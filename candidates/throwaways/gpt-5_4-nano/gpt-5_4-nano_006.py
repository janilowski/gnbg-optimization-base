# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a simple, robust black-box minimization algorithm using
# a mixture of (1) random global sampling for exploration and (2) a local
# coordinate/gaussian search around the best point found so far for exploitation.
# It operates only through function evaluations and respects the provided budget.
# Search state: Tracks current best point x_best and its objective value y_best,
# along with a step-size (sigma) that is adapted based on recent improvements.
# Candidate generation: Proposes candidates by either sampling uniformly within
# the bounds (exploration) or perturbing the current best using Gaussian noise
# scaled by sigma and also performing coordinate-wise +/- steps.
# Selection and replacement: Evaluates candidates, keeps the best one, and updates
# x_best/y_best when an improvement is found.
# Adaptation: Uses a multiplicative step-size schedule: when improvements occur,
# sigma is reduced (refine); when not, sigma is enlarged (escape local traps),
# bounded to avoid numerical issues.
# Exploration mechanisms: Random uniform sampling of several points early and
# occasional stochastic proposals during the run, proportionally decreasing with
# remaining budget.
# Exploitation mechanisms: Coordinate-wise probing and local Gaussian proposals
# around x_best to efficiently improve near the incumbent.
# Boundary handling: Every candidate is clipped to the feasible box [lb, ub].
# Budget strategy: Strictly counts evaluations and never exceeds the budget; the
# number of proposals is chosen to use at most the remaining evaluations.
# Closest known influences: Inspired by basic derivative-free strategies combining
# global random search with local refinement (coordinate search / evolution-strategy-like
# sigma adaptation), implemented compactly without external dependencies.
# Novelty or unusual aspects: Combines both uniform exploration and two local
# proposal types (coordinate +/- and gaussian perturbations) with a simple
# success-based sigma adaptation; designed to remain stable across dimensions.
# Failure modes: If the objective is extremely deceptive or the budget is too
# small, the algorithm may not locate the global optimum; clipping can also
# bias search near boundaries for strongly constrained problems.
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

        # --- Bounds reading (supports multiple common conventions) ---
        lb = None
        ub = None

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            lb = np.asarray(getattr(b, "lb"), dtype=float)
            ub = np.asarray(getattr(b, "ub"), dtype=float)
        else:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub.")

        if lb.shape == () and dim != 1:
            lb = np.full(dim, float(lb))
        if ub.shape == () and dim != 1:
            ub = np.full(dim, float(ub))

        lb = lb.reshape(dim)
        ub = ub.reshape(dim)

        # Safety: if bounds are inverted or equal, handle gracefully.
        width = ub - lb
        width = np.where(width == 0.0, 1.0, width)  # avoid division by zero in heuristics

        def clip(x):
            # Clip robustly and preserve shape.
            return np.minimum(ub, np.maximum(lb, x))

        # Evaluation wrapper with strict budget enforcement.
        evals_used = 0
        best_x = None
        best_y = None

        def eval_once(x):
            nonlocal evals_used, best_x, best_y
            if evals_used >= budget:
                return best_y
            y = func(x)
            evals_used += 1
            # Objective is minimization
            if best_y is None or y < best_y:
                best_y = float(y)
                best_x = np.array(x, dtype=float, copy=True)
            return y

        # If budget is zero, return something deterministic (best_x remains None).
        # But per benchmark conventions, budget should be >= 1.
        if budget <= 0:
            # Return midpoint clipped.
            x0 = clip((lb + ub) / 2.0)
            return x0, float(func(x0)) if budget == 0 else (x0, np.inf)

        # --- Initialization: start from a few random points then exploit ---
        # Choose how many initial samples: min(20, budget) but scaled with dimension.
        # More dimensions -> more initial exploration.
        k_init = min(budget, max(2, int(round(2 + 0.5 * dim))))
        for _ in range(k_init):
            x = lb + np.random.rand(dim) * (ub - lb)
            eval_once(x)

        # Heuristic step-size: fraction of box width.
        # Using average width gives stable scaling across dimensions.
        avg_w = float(np.mean(np.abs(ub - lb))) if dim > 0 else 1.0
        sigma = 0.3 * (avg_w if avg_w > 0 else 1.0)
        sigma = max(sigma, 1e-12)

        # Track successes to adapt sigma.
        successes = 0
        fails = 0

        # --- Main loop: mixed exploration + exploitation until budget is exhausted ---
        while evals_used < budget:
            remaining = budget - evals_used

            # Decreasing exploration probability as we approach budget end.
            # (Early: explore more, later: exploit more.)
            t = evals_used / max(1, budget)
            explore_prob = max(0.15, 0.55 * (1.0 - t))

            improved_in_iter = False

            # Decide candidate batches: small, efficient, and budget-aware.
            # Number of proposals per iteration.
            batch_size = min(1 + dim // 2 + 2, remaining)  # keep it modest

            candidates = []

            if np.random.rand() < explore_prob:
                # Exploration: uniform random samples
                # (batch_size candidates)
                for _ in range(batch_size):
                    x = lb + np.random.rand(dim) * (ub - lb)
                    candidates.append(x)
            else:
                # Exploitation: coordinate +/- probes + gaussian perturbations
                # Create up to batch_size candidates.
                # Coordinate order randomized each iteration.
                axes = np.random.permutation(dim)
                # Coordinate step based on sigma and relative width
                # per-coordinate scaling uses width magnitude to adapt to varying bounds.
                rel = np.abs(ub - lb)
                rel = np.where(rel == 0.0, 1.0, rel)

                # How many coordinate probes vs gaussian perturbations?
                # Ensure we don't exceed batch_size.
                n_coord = min(dim, max(2, batch_size // 2))
                step = sigma / np.sqrt(rel)  # smaller where bounds are tight

                x0 = best_x
                # Coordinate +/- around x0
                for i in range(n_coord):
                    ax = axes[i]
                    dx = np.zeros(dim, dtype=float)
                    dx[ax] = step[ax]
                    candidates.append(clip(x0 + dx))
                    if len(candidates) < batch_size:
                        candidates.append(clip(x0 - dx))
                    if len(candidates) >= batch_size:
                        break

                # Fill remaining with gaussian perturbations
                while len(candidates) < batch_size:
                    noise = np.random.randn(dim) * sigma
                    x = clip(x0 + noise)
                    candidates.append(x)

            # Evaluate candidates and track improvement
            y_before = best_y
            # Evaluate sequentially; budget is enforced in eval_once.
            for x in candidates:
                if evals_used >= budget:
                    break
                prev_best = best_y
                eval_once(x)
                if prev_best is not None and best_y is not None and best_y < prev_best:
                    improved_in_iter = True

            # Adapt sigma based on success/failure.
            if improved_in_iter:
                successes += 1
                fails = 0
                # Reduce sigma to refine around new best.
                sigma *= 0.85
            else:
                fails += 1
                successes = 0
                # Mildly increase sigma to escape stagnation.
                sigma *= 1.10

            # Bound sigma to a reasonable range using average width.
            max_sigma = 0.9 * (avg_w if avg_w > 0 else 1.0)
            min_sigma = 1e-12
            sigma = float(np.clip(sigma, min_sigma, max_sigma))

            # Early termination if sigma becomes extremely small and no improvement
            if fails >= 8 and sigma <= min_sigma * 10:
                break

        # Ensure best_x exists; if not (possible only if budget == 0 but handled above)
        if best_x is None:
            best_x = clip((lb + ub) / 2.0)
            best_y = float(func(best_x))

        return best_x, best_y
