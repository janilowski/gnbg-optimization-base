# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, budget-aware black-box minimizer.
# It maintains a single incumbent best solution and repeatedly generates new
# candidate points by mixing a global exploratory step with local exploitation
# around the current best. The search operates directly in continuous space
# and supports arbitrary dimensionality.
#
# Search state: The algorithm tracks (1) the incumbent best_x and best_y,
# (2) the remaining evaluation budget, and (3) an adaptive step size that
# shrinks when improvements are found and grows slightly otherwise.
#
# Candidate generation: Each iteration draws multiple trial points by:
# - Sampling a random direction and step length for exploration.
# - Sampling a Gaussian perturbation around the current best for exploitation.
# - Optionally combining the best with a second random point via a small
#   linear interpolation (a lightweight diversification move).
#
# Selection and replacement: All trial points are evaluated (carefully staying
# within the remaining budget). The best among them replaces the incumbent if
# it improves; otherwise the incumbent remains.
#
# Adaptation: If an improvement occurs, step size is reduced (more local).
# If no improvement occurs, step size is increased modestly (more global).
# The step size is bounded to avoid stalling or overshooting.
#
# Exploration mechanisms: Random directions with uniform step lengths and
# occasional best–random interpolation encourage movement beyond local basins.
#
# Exploitation mechanisms: Gaussian perturbations around the incumbent with a
# step size that adapts to recent progress.
#
# Boundary handling: Candidate points are clipped to the feasible box derived
# from func.lower/func.upper or func.bounds.lb/func.bounds.ub.
#
# Budget strategy: The algorithm never evaluates more than `budget` total
# points by tracking the number of evaluations. It uses a small batch size
# early on and then falls back to single evaluations when the remaining budget
# is tight.
#
# Closest known influences: General-purpose evolution-strategy style search:
# incumbent-based sampling with adaptive step size and bounded box handling,
# inspired by simple (1+λ)-ES / CEM-like heuristics but kept lightweight.
#
# Novelty or unusual aspects: Uses an evaluation-efficient mixed candidate
# set (explore + exploit + interpolation) while dynamically adjusting batch
# size based on remaining budget.
#
# Failure modes: For extremely rugged landscapes or pathological bounds,
# clipping may reduce effective diversity. If the optimum is very narrow,
# step-size adaptation may converge prematurely; however, the conservative
# step-size growth on non-improvement helps mitigate this.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim

        # --- Read bounds robustly ---
        lb, ub = None, None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            # Some harnesses may use lb/ub or lower/upper style inside bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            elif hasattr(b, "lower") and hasattr(b, "upper"):
                lb = np.asarray(b.lower, dtype=float)
                ub = np.asarray(b.upper, dtype=float)

        if lb is None or ub is None:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub")

        lb = np.broadcast_to(lb, (dim,)).copy()
        ub = np.broadcast_to(ub, (dim,)).copy()
        span = ub - lb
        # Avoid degenerate span issues
        span_safe = np.where(span > 0, span, 1.0)

        def clip(x: np.ndarray) -> np.ndarray:
            return np.minimum(ub, np.maximum(lb, x))

        def eval_one(x: np.ndarray) -> float:
            return float(func(x))

        # --- Budget-aware evaluation tracking ---
        evals = 0
        if self.budget <= 0:
            # No evaluations possible; return a feasible point anyway
            x0 = clip(lb + 0.5 * span_safe)
            return x0, float("inf")

        # Initialize incumbent: start with a midpoint, then optionally one random point if budget allows
        x_best = clip(lb + 0.5 * span_safe)
        y_best = eval_one(x_best)
        evals += 1

        if self.budget - evals > 0:
            # one random initial point
            x_r = clip(lb + np.random.rand(dim) * span_safe)
            y_r = eval_one(x_r)
            evals += 1
            if y_r < y_best:
                x_best, y_best = x_r, y_r

        # Adaptive step size: fraction of domain
        # Start moderately sized; adjust based on improvements.
        # Use geometric scaling based on dimension to be reasonably invariant.
        base_scale = 0.25
        sigma = base_scale * np.mean(span_safe)
        sigma_min = 1e-12
        sigma_max = 1.0 * np.mean(span_safe) if np.mean(span_safe) > 0 else 1.0

        rng = np.random

        # Helper: how many candidates to evaluate this round without exceeding budget
        def choose_batch(remaining: int) -> int:
            # Small batch for overhead-free exploration, larger batch when budget allows.
            # Keep it bounded to stay efficient.
            if remaining >= 16:
                return 8
            if remaining >= 8:
                return 5
            if remaining >= 4:
                return 3
            return 1

        # --- Main loop ---
        # Each iteration generates a small batch of candidates:
        # explore (random directions), exploit (Gaussian around best), and mix (interpolation).
        while evals < self.budget:
            remaining = self.budget - evals
            batch = choose_batch(remaining)

            # Allocate candidates
            # We will always include at least one exploit and one explore when possible.
            # For batch=1, it's exploit-focused.
            candidates = []
            # Exploit candidate(s)
            n_exploit = 1 if batch == 1 else max(1, batch // 2)
            n_explore = batch - n_exploit

            # --- Exploration ---
            for _ in range(n_explore):
                # Random direction normalized to unit length
                d = rng.normal(size=dim)
                norm = np.linalg.norm(d)
                if not np.isfinite(norm) or norm == 0.0:
                    d = rng.normal(size=dim)
                    norm = np.linalg.norm(d)
                d = d / max(norm, 1e-12)
                # Step length uniformly in a reasonable range relative to sigma
                step = (0.25 + 0.75 * rng.random()) * sigma
                x = x_best + d * step
                candidates.append(clip(x))

            # --- Exploitation ---
            for _ in range(n_exploit):
                noise = rng.normal(size=dim)
                x = x_best + sigma * noise
                candidates.append(clip(x))

            # --- Optional interpolation diversification ---
            # If we have extra capacity in the batch, replace one candidate with a blend move.
            if batch >= 3:
                # Create one more diverse blend candidate by interpolating between best and a random point
                x_other = clip(lb + rng.random(dim) * span_safe)
                t = 0.1 + 0.8 * rng.random()  # biased away from extremes
                x_blend = clip((1.0 - t) * x_best + t * x_other)
                candidates[-1] = x_blend

            # Evaluate and select best among candidates within budget
            # Note: batch is chosen to never exceed remaining, but keep safe.
            best_y_local = y_best
            best_x_local = x_best
            for x in candidates:
                if evals >= self.budget:
                    break
                y = eval_one(x)
                evals += 1
                if y < best_y_local:
                    best_y_local = y
                    best_x_local = x

            improved = best_y_local < y_best
            if improved:
                x_best, y_best = best_x_local, best_y_local
                # Shrink sigma when progress is made (focus locally)
                sigma = max(sigma_min, sigma * 0.72)
            else:
                # Expand slightly when stuck (encourage exploration)
                sigma = min(sigma_max, sigma * 1.10)

            # Additional safeguard: if sigma becomes extremely small, re-inflate to avoid stagnation
            if sigma <= sigma_min * 10:
                sigma = min(sigma_max, 0.5 * sigma_max)

        return x_best, y_best
