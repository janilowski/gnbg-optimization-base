# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# inspired by a two-stage approach: (1) global-ish sampling to initialize a
# promising region, then (2) iterative local refinement using coordinate-wise
# probes and a shrinking step-size schedule.
# Search state: The algorithm maintains the current best point x_best and
# its objective value y_best, along with a current probe step-size sigma.
# Candidate generation: Each iteration generates a small set of candidate points
# by moving from the current best along coordinate directions (+/- sigma) and
# also includes occasional “random” candidates scaled by sigma to maintain
# robustness across dimensions.
# Selection and replacement: After evaluating candidates, the best feasible
# (within provided bounds) candidate replaces the current best. If no
# improvement occurs, the algorithm keeps the best so far and reduces sigma.
# Adaptation: sigma shrinks when iterations fail to improve, and expands
# slightly after improvements (capped) to help escape overly conservative steps.
# Exploration mechanisms: Random candidates are injected with decreasing
# probability as sigma shrinks; additionally, coordinate probing covers both
# directions deterministically.
# Exploitation mechanisms: Coordinate-wise probes around the best point provide
# local gradient-free refinement.
# Boundary handling: Candidate points are clipped to the provided bounds.
# Budget strategy: The algorithm never exceeds the evaluation budget. It tracks
# the number of objective evaluations and uses a pre-planned number of probes
# per iteration while stopping once the budget is exhausted.
# Closest known influences: The design loosely follows ideas from pattern search
# and CMA-ES-like “best-centered” local sampling, but remains much simpler and
# budget-aware.
# Novelty or unusual aspects: It adaptively allocates evaluation effort between
# initial sampling and local coordinate probing while maintaining strict budget
# compliance with minimal overhead.
# Failure modes: If the objective is very noisy or deceptive, coordinate probing
# may stagnate; the algorithm mitigates this with random candidates and sigma
# adaptation, but worst-case performance is not guaranteed.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return a zero vector (undefined objective).
            return np.zeros(dim, dtype=float), float("inf")

        # Read bounds
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        # Fallback to infinite bounds if not provided
        if lb is None or ub is None:
            lb = np.full(dim, -np.inf, dtype=float)
            ub = np.full(dim, np.inf, dtype=float)

        # Ensure shapes
        if lb.shape == () or lb.size == 1:
            lb = np.full(dim, float(lb), dtype=float)
        if ub.shape == () or ub.size == 1:
            ub = np.full(dim, float(ub), dtype=float)

        # Clip helper (handles +/-inf safely)
        def clip_to_bounds(x):
            return np.minimum(np.maximum(x, lb), ub)

        # Evaluation wrapper with strict budget accounting
        eval_count = 0
        best_x = None
        best_y = float("inf")

        def eval_x(x):
            nonlocal eval_count, best_x, best_y
            y = float(func(x))
            eval_count += 1
            if y < best_y:
                best_y = y
                best_x = np.array(x, copy=True)
            return y

        # Determine a reasonable initial sigma from bounds
        span = ub - lb
        finite_span = np.isfinite(span)
        if np.any(finite_span):
            base_span = np.median(span[finite_span])
            if not np.isfinite(base_span) or base_span <= 0:
                base_span = 1.0
        else:
            base_span = 1.0
        # sigma scale: fraction of range (or 1.0 if range unknown)
        sigma = 0.2 * float(base_span)

        # Initial design: sample a small batch uniformly to find a good seed.
        # Keep it small to reserve budget for local refinement.
        init_budget = max(1, min(budget // 5 + 1, 12))
        init_budget = min(init_budget, budget)

        # Random initial points
        for _ in range(init_budget):
            r = np.random.rand(dim)
            x = lb + r * (ub - lb) if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)) else np.random.randn(dim)
            if np.any(np.isfinite(lb) | np.isfinite(ub)):
                x = clip_to_bounds(x)
            eval_x(x)
            if eval_count >= budget:
                break

        # If budget remains, do iterative coordinate probing around best_x
        if eval_count < budget:
            # Number of coordinate probes per iteration:
            # - For larger dimensions, probe only a subset of coordinates per iter.
            # - For small dimensions, probe all.
            max_iters = 10_000  # will be cut by budget anyway
            step_shrink = 0.85
            step_expand = 1.15
            sigma_min = 1e-12

            # Subset size schedule
            if dim <= 8:
                subset = dim
            else:
                subset = max(3, int(round(np.sqrt(dim) * 2)))

            for _ in range(max_iters):
                if eval_count >= budget:
                    break

                # Choose coordinates to probe: include a few random coordinates each iteration
                # while ensuring we always include deterministic candidates.
                if subset >= dim:
                    coords = np.arange(dim)
                else:
                    coords = np.random.choice(dim, size=subset, replace=False)

                # Build candidate list: +sigma and -sigma along chosen coordinates
                # plus occasional random exploration.
                candidates = []
                # Always include coordinate probes
                for j in coords:
                    e = np.zeros(dim, dtype=float)
                    e[j] = 1.0
                    candidates.append(clip_to_bounds(best_x + sigma * e))
                    candidates.append(clip_to_bounds(best_x - sigma * e))

                # Exploration: with decreasing probability, add a few random points
                remaining = budget - eval_count
                # Keep per-iteration evaluations bounded and budget-aware
                # Evaluate at most remaining.
                # We'll cap the random candidates to avoid explosion in candidate count.
                p_explore = 0.35 * (sigma / (0.2 * base_span + 1e-12))
                p_explore = float(np.clip(p_explore, 0.02, 0.35))
                if np.random.rand() < p_explore and remaining > 0:
                    # Add up to 2 random perturbations
                    n_rand = 1 + (np.random.rand() < 0.5)
                    for _k in range(n_rand):
                        z = np.random.randn(dim)
                        # Scale by sigma but normalize to keep magnitude comparable across dim
                        norm = np.linalg.norm(z) + 1e-12
                        z = z / norm
                        x = clip_to_bounds(best_x + (0.5 * sigma) * z)
                        candidates.append(x)

                # Evaluate candidates with strict remaining budget
                improved = False
                # Shuffle candidates to avoid always evaluating a fixed pattern first
                if len(candidates) > 1:
                    perm = np.random.permutation(len(candidates))
                    candidates = [candidates[i] for i in perm]

                for x in candidates:
                    if eval_count >= budget:
                        break
                    # Avoid exact duplicates when possible (cheap check)
                    # (Not perfect but harmless.)
                    # If best_x exists, skip if too close to it.
                    if best_x is not None and np.all(np.isfinite(x)) and np.all(np.isfinite(best_x)):
                        if np.linalg.norm(x - best_x) <= 1e-15:
                            continue
                    y_before = best_y
                    eval_x(x)
                    if best_y < y_before:
                        improved = True

                # Adapt sigma based on improvement
                if improved:
                    sigma = min(sigma * step_expand, 2.0 * (ub - lb).max(initial=1.0))
                else:
                    sigma *= step_shrink

                if sigma < sigma_min:
                    break

        # Ensure best_x is set; if not, evaluate one point (shouldn't happen unless budget=0)
        if best_x is None:
            x0 = np.zeros(dim, dtype=float)
            x0 = clip_to_bounds(x0)
            if eval_count < budget:
                eval_x(x0)
            else:
                x0 = clip_to_bounds(x0)

        return np.array(best_x, copy=True), float(best_y)
