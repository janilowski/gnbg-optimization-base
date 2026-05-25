# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm (gradient-free)
# using a population of candidate points, noisy "local" refinement with coordinate
# probing, and a global exploration phase based on a shrinking random step.
# Search state: Maintains a current best point (best_x, best_y) and a step-size
# (sigma) that shrinks when improvements are found and expands slightly otherwise.
# Candidate generation: Each iteration creates a set of candidates by:
#   1) Sampling around the current best point with Gaussian noise scaled by sigma.
#   2) Performing coordinate-wise probing (finite-difference-like) along a few
#      directions using random coordinate axes to estimate which way is better.
# Selection and replacement: Evaluates all candidates, keeps the best, and replaces
# the global best if any candidate improves the objective.
# Adaptation: Uses a success-based controller: if the best improves, sigma is
# reduced; if no improvement occurs for several iterations, sigma is increased.
# Exploration mechanisms: Global random sampling around best with occasional
# larger step during stagnation, plus coordinate probing to escape flat regions.
# Exploitation mechanisms: Gaussian sampling tightly around best and local
# coordinate probing to refine along promising directions.
# Boundary handling: Candidate points are clamped to the provided bounds after
# every generation to ensure feasibility.
# Budget strategy: The total number of objective evaluations is capped by the
# provided budget. The algorithm stops immediately when the remaining budget is
# exhausted.
# Closest known influences: Combines ideas reminiscent of CMA-ES-like success
# adaptation (simplified), evolution strategies sampling, and coordinate search
# refinement (without gradients).
# Novelty or unusual aspects: Mixes a small population ES-style sampling with
# a lightweight coordinate probing step, while using only the Python stdlib and
# numpy and enforcing strict evaluation accounting.
# Failure modes: If the objective is extremely noisy or highly ill-conditioned,
# sigma adaptation may oscillate; clamping can cause many repeated boundary
# evaluations; very small budgets may lead to limited search.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import math
import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from supported interface patterns ----
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

        if lb is None or ub is None:
            raise AttributeError(
                "Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )

        # Ensure correct shapes
        lb = np.broadcast_to(lb, (dim,)).astype(float, copy=False)
        ub = np.broadcast_to(ub, (dim,)).astype(float, copy=False)

        # Handle degenerate bounds
        span = ub - lb
        span = np.where(span == 0.0, 1.0, span)  # avoid zeros in sigma calculations

        def clamp(x):
            # x: (dim,) or (n, dim)
            return np.minimum(np.maximum(x, lb), ub)

        evals = 0
        best_x = None
        best_y = None

        # Objective wrapper with strict evaluation accounting
        def eval_x(x):
            nonlocal evals, best_x, best_y
            y = func(x)
            evals += 1
            if best_y is None or y < best_y:
                best_y = float(y)
                best_x = np.array(x, dtype=float, copy=True)
            return float(y)

        if budget <= 0:
            # No evaluations allowed: return a feasible point deterministically
            mid = (lb + ub) / 2.0
            return clamp(mid), float("inf")

        # ---- Initialize with a few random points plus midpoint ----
        rng = np.random

        mid = (lb + ub) / 2.0

        # Use up to K initial evaluations without exceeding budget.
        K = min(4, budget)  # small constant population
        # Start with midpoint and random points for diversity
        candidates0 = [mid]
        for _ in range(max(0, K - 1)):
            r = rng.random(dim)
            x = lb + r * (ub - lb)
            candidates0.append(x)

        for x in candidates0:
            if evals >= budget:
                break
            eval_x(clamp(x))

        if evals >= budget:
            return best_x, best_y

        # ---- Search parameters ----
        # Initial sigma based on bounds span (robust across scales).
        sigma = 0.3 * np.mean(np.abs(span))
        sigma = max(sigma, 1e-12)

        # Iteration budget planning
        remaining = budget - evals
        # We will adaptively compute evaluations per iteration.
        max_iters = max(1, int(math.ceil(remaining / max(4, dim + 2))))
        stagnation = 0

        # Number of candidates per iteration (keep modest)
        # Also dependent on dimension: more dimensions => a bit more exploration.
        base_pop = min(10, max(4, dim + 1))
        # Coordinate probing: probe a small subset of axes to keep budget sane.
        coord_probes = min(dim, max(2, dim // 2))

        # ---- Main loop ----
        for it in range(max_iters):
            if evals >= budget:
                break

            remaining = budget - evals
            # Make sure we don't overshoot the budget.
            # We'll evaluate at most pop + 2*coord_probes candidates, but cap by remaining.
            pop = min(base_pop, remaining)

            # Candidate list
            cand = []

            # 1) ES-style sampling around best (exploitation/exploration)
            # Draw in batch then slice to desired count.
            if best_x is None:
                x0 = mid
            else:
                x0 = best_x

            # Gaussian steps; we bias toward exploitation by centering at best.
            # Occasional larger exploration if stagnating.
            explore_mult = 1.0 + (0.75 if stagnation >= 2 else 0.0)
            s = sigma * explore_mult

            # Create more than needed then slice (cheaper than branching)
            n_pre = min(2 * pop + 2, remaining)  # limited to reasonable size
            z = rng.standard_normal((n_pre, dim))
            xs = x0 + s * z
            xs = clamp(xs)
            cand.extend(xs.tolist())

            # 2) Coordinate probing (lightweight local search)
            if len(cand) < pop:
                # Pick random unique axes to probe
                axes = rng.choice(dim, size=coord_probes, replace=False)
                # Step length along axes, relative to sigma and bounds span
                # Use both sign directions.
                # Use a step that is neither too tiny nor too large.
                step = 0.2 * sigma
                step = max(step, 1e-12)

                for ax in axes:
                    if len(cand) >= pop:
                        break
                    e = np.zeros(dim, dtype=float)
                    e[ax] = 1.0
                    x_plus = clamp(x0 + step * e)
                    x_minus = clamp(x0 - step * e)
                    # Only add if not identical to existing best, but still safe to evaluate.
                    cand.append(x_plus.tolist())
                    if len(cand) < pop:
                        cand.append(x_minus.tolist())

            # Slice to pop candidates if oversized
            if len(cand) > pop:
                cand = cand[:pop]

            # Evaluate candidates, but ensure strict budget cap
            improved_before = best_y
            for x in cand:
                if evals >= budget:
                    break
                eval_x(np.asarray(x, dtype=float))

            # Adapt sigma based on improvement
            if best_y is not None and improved_before is not None and best_y < improved_before - 1e-15:
                # Improvement: shrink sigma for exploitation
                sigma *= 0.85
                stagnation = 0
            else:
                # No improvement: increase sigma slightly to explore
                stagnation += 1
                sigma *= 1.10
                # Prevent sigma from collapsing to zero or exploding wildly.
                sigma = float(np.clip(sigma, 1e-12, 0.9 * np.mean(np.abs(span)) + 1e-12))

            # Mild early stopping if sigma is extremely small (likely stuck on boundaries)
            if sigma <= 1e-11 and stagnation >= 3:
                break

        # If for some reason best_x not set (shouldn't happen if budget>0), set midpoint.
        if best_x is None:
            best_x = clamp(mid)
            # If budget allowed, at least evaluate once; otherwise return inf.
            if evals < budget:
                best_y = float(func(best_x))
            else:
                best_y = float("inf")

        return best_x, best_y
