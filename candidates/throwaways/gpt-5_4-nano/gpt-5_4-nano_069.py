# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimizer
# using a population-based random-search with iterative shrinking neighborhoods.
# It targets the best observed point, progressively refining the search radius
# around the current best while maintaining a small amount of global exploration.
# Search state: The algorithm maintains a current best point x_best and its
# objective value y_best, plus a per-iteration search radius (step size).
# Candidate generation: Each iteration samples a small set of candidate points
# around x_best by adding Gaussian noise scaled by the current radius and the
# per-dimension span. Additionally, it may include one purely random candidate
# to escape stagnation.
# Selection and replacement: All evaluated candidates are compared to y_best;
# if a candidate improves the objective, it becomes the new x_best.
# Adaptation: The radius shrinks when improvements are found, and (mildly)
# shrinks or is reset when improvements stall, balancing exploration/exploitation.
# Exploration mechanisms: A probability of adding a global random candidate
# and occasional radius resets help the search avoid local traps.
# Exploitation mechanisms: Most candidates are drawn from a Gaussian centered at
# x_best with shrinking radius, intensifying search near the best point.
# Boundary handling: Candidates are clipped to the provided bounds after each
# perturbation.
# Budget strategy: The evaluation budget is split across iterations with
# careful tracking of remaining evaluations. The algorithm never exceeds the
# budget provided by the harness.
# Closest known influences: Inspired by simple evolution strategies / CMA-like
# behavior but intentionally lightweight: isotropic Gaussian sampling plus
# success-based step-size adaptation.
# Novelty or unusual aspects: Uses a success counter to decide how aggressively
# to shrink/reset the search radius while ensuring budget safety in a compact
# implementation.
# Failure modes: If the objective is extremely rugged with very narrow basins,
# the isotropic sampling may struggle; budget exhaustion can also limit progress
# before reaching a good region.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # ---- Read bounds from func ----
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub")

        n = self.dim
        lb = np.broadcast_to(lb, (n,)).copy()
        ub = np.broadcast_to(ub, (n,)).copy()

        # Ensure valid bounds
        span = ub - lb
        span = np.where(span > 0, span, 1.0)  # avoid degenerate span

        # ---- Budget management ----
        max_evals = max(1, int(self.budget))
        evals = 0

        def eval_x(x):
            nonlocal evals
            if evals >= max_evals:
                # Should not happen with careful budgeting; fail-safe.
                return np.inf
            evals += 1
            return float(func(x))

        # ---- Initial sampling ----
        rng = np.random
        # Try a handful of initial points; keep it budget-safe.
        init_tries = min(max_evals, 5)
        # Use one center point plus random points (if budget allows).
        candidates = []
        x_center = lb + 0.5 * span
        candidates.append(x_center)
        for _ in range(init_tries - 1):
            t = rng.rand(n)
            candidates.append(lb + t * span)

        best_x = None
        best_y = np.inf
        for x in candidates:
            y = eval_x(np.clip(x, lb, ub))
            if y < best_y:
                best_y = y
                best_x = np.array(x, dtype=float, copy=True)

        if evals >= max_evals:
            return best_x, best_y

        # ---- Iteration plan ----
        # Use a modest number of iterations; per-iteration sample a small batch.
        # The batch size grows mildly with dimension, but is capped.
        batch = int(np.clip(4 + n // 10, 4, 20))
        # Remaining budget after initialization
        rem = max_evals - evals
        iters = max(1, rem // batch)
        # If rem < batch, do one more iteration with fewer samples.
        if iters < 1:
            iters = 1

        # Radius initial: fraction of the domain span.
        radius = 0.25
        # Track improvements to adapt radius.
        no_improve = 0
        success_streak = 0

        # Probability of global exploration
        p_global = 0.15 + min(0.2, 0.02 * np.log1p(n))

        for _ in range(iters):
            if evals >= max_evals:
                break

            # Determine how many evaluations we can still spend this iteration
            remaining = max_evals - evals
            k = min(batch, remaining)

            # Adapt radius based on stagnation
            if no_improve >= 6:
                # Reset a bit to re-explore if we've been stuck
                radius = min(0.5, radius * 1.5)
                no_improve = 0
                success_streak = 0

            # Effective step scale per dimension
            step = radius * span

            improved_this_iter = False
            for j in range(k):
                # Decide candidate type: exploitation around best or exploration
                if rng.rand() < p_global and j == 0:
                    # Global random candidate
                    x = lb + rng.rand(n) * span
                else:
                    # Gaussian step around best
                    # Use isotropic noise; optionally vary magnitude slightly.
                    noise = rng.randn(n)
                    # Scale noise by a log-uniform-ish factor to add mild diversity
                    mag = np.exp(rng.uniform(-0.35, 0.35))
                    x = best_x + (step * mag) * noise

                # Boundary handling: clip
                x = np.clip(x, lb, ub)

                y = eval_x(x)
                if y < best_y:
                    best_y = y
                    best_x = x
                    improved_this_iter = True

            if improved_this_iter:
                # Success: shrink radius to exploit
                success_streak += 1
                no_improve = 0
                radius *= (0.7 ** min(2, success_streak))
                # Keep radius from collapsing too quickly
                radius = max(radius, 1e-6)
            else:
                # Failure: mild shrink, but not too aggressive to keep exploration
                no_improve += 1
                success_streak = 0
                radius *= 0.9
                radius = max(radius, 1e-9)

        return best_x, best_y
