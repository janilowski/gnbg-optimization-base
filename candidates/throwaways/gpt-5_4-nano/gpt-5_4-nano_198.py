# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, budget-safe black-box optimizer for
# minimization. It uses a population of candidate solutions sampled around a
# moving “center” (best-so-far) and progressively shrinks the search radius.
# Search state: The algorithm tracks the current best point and value, the
# remaining evaluation budget, and a step-size (radius) controlling how far new
# candidates are sampled from the center.
# Candidate generation: Each iteration samples a small population of points by
# drawing Gaussian perturbations of the current center scaled by the step-size.
# It also includes a few structured directions (axis-aligned perturbations) to
# improve robustness in different dimensions.
# Selection and replacement: After evaluating all candidates in an iteration, the
# best candidate replaces the current center if it improves the best-so-far value.
# The center otherwise stays as the global best to maintain correctness under
# noisy or discontinuous objectives.
# Adaptation: The step-size shrinks when improvements are found, and otherwise
# modestly shrinks as well (with occasional resets to avoid stagnation).
# Exploration mechanisms: Larger step-size early on and occasional larger
# perturbations (reset) help escape local minima.
# Exploitation mechanisms: Gaussian sampling focused around the best-so-far point
# plus contraction of the step-size drives exploitation.
# Boundary handling: Candidate points are clipped to provided bounds before
# evaluation; this guarantees feasibility.
# Budget strategy: The total number of objective evaluations is capped by the
# provided budget. The algorithm uses a fixed per-iteration population size and
# adjusts the number of iterations/candidates to never exceed budget.
# Closest known influences: The behavior resembles a simplified evolution strategy
# / CMA-like "best-centre" sampling with step-size adaptation, implemented
# without advanced covariance learning.
# Novelty or unusual aspects: Uses a small mix of isotropic Gaussian sampling and
# axis perturbations, plus a deterministic budget accounting and safe boundary
# clipping, designed to be compact and reliable across dimensions.
# Failure modes: If the objective is extremely deceptive or very noisy, step-size
# contraction can cause premature convergence; the occasional reset mitigates this.
# If bounds are very tight or dimension is high, clipping may reduce effective
# search movement. The algorithm still honors budget and returns the best found.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _extract_bounds(self, func):
        # Support func.lower/func.upper or func.bounds.lb/ub (per requirements).
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            else:
                raise AttributeError("func.bounds must provide lb and ub")
        else:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub")

        lb = np.reshape(lb, (-1,))
        ub = np.reshape(ub, (-1,))
        if lb.size != self.dim or ub.size != self.dim:
            # Allow scalar bounds repeated across dim.
            if lb.size == 1:
                lb = np.full(self.dim, float(lb[0]))
            if ub.size == 1:
                ub = np.full(self.dim, float(ub[0]))
        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError("Bounds dimensionality does not match dim")
        if np.any(ub < lb):
            raise ValueError("Invalid bounds: ub must be >= lb elementwise")
        return lb, ub

    def __call__(self, func):
        lb, ub = self._extract_bounds(func)

        # Budget safety: number of evaluations including initial evaluations.
        budget = max(1, int(self.budget))
        dim = self.dim

        evals = 0

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        # Determine a reasonable initial center: random point within bounds.
        # Harness sets numpy seed for reproducibility.
        center = lb + np.random.rand(dim) * (ub - lb)
        center = clip(center)

        best_x = center.copy()
        best_y = float(func(best_x))
        evals += 1
        # If budget is 1, return immediately.
        if evals >= budget:
            return best_x, best_y

        # Initial step-size: fraction of box size (robust if some dimensions have zero width).
        box = ub - lb
        # Use robust nonzero scale; if all zero, the function is constant in feasible set.
        nonzero = box > 0
        if np.any(nonzero):
            scale = 0.25 * np.median(box[nonzero])
            # If median is 0 due to tiny numbers, fallback.
            if not np.isfinite(scale) or scale <= 0:
                scale = 0.1 * float(np.max(box))
        else:
            scale = 0.0

        # Population size per iteration: keep small and dimension-aware.
        # Ensure at least 2 candidates in each iteration when possible.
        base_pop = 6
        pop = int(min(budget - evals, max(2, min(10, base_pop + dim // 4))))
        # If dim is very high, still keep manageable.
        pop = max(2, min(pop, 16))

        # Step-size adaptation parameters.
        sigma = scale if scale > 0 else 1.0
        shrink_on_improve = 0.65
        shrink_on_no_improve = 0.9
        reset_prob = 0.08 if dim >= 2 else 0.12
        reset_factor = 2.2

        # Axis perturbation magnitude relative to sigma.
        axis_mag = 0.35

        # Main loop: each iteration consumes up to pop evaluations.
        while evals < budget:
            remaining = budget - evals
            k = pop if remaining >= pop else remaining

            # Generate candidates around the center.
            # Candidates are clipped to bounds before evaluation.
            candidates = np.empty((k, dim), dtype=float)

            # Mix strategy:
            # - Most candidates: isotropic Gaussian around center.
            # - A few: axis-aligned perturbations to help explore along coordinate directions.
            # Choose number of axis candidates based on dimension.
            axis_count = min(k, max(1, k // 5))
            gauss_count = k - axis_count

            # Gaussian candidates
            if gauss_count > 0:
                noise = np.random.randn(gauss_count, dim)
                candidates[:gauss_count] = center + sigma * noise

            # Axis-aligned candidates
            if axis_count > 0:
                idxs = np.random.randint(0, dim, size=axis_count)
                # Random signs for axis perturbations
                signs = np.where(np.random.rand(axis_count) < 0.5, -1.0, 1.0)
                # Different magnitudes to avoid symmetry
                mags = axis_mag * (0.5 + np.random.rand(axis_count))
                # Start from center, then perturb one coordinate
                candidates[gauss_count:] = center
                for i, (j, s, m) in enumerate(zip(idxs, signs, mags), start=gauss_count):
                    candidates[i, j] = candidates[i, j] + s * m * sigma

            # Clip candidates
            candidates = clip(candidates)

            # Evaluate candidates
            ys = np.empty(k, dtype=float)
            for i in range(k):
                y = float(func(candidates[i]))
                ys[i] = y
                evals += 1
                if evals >= budget and i < k - 1:
                    # In practice, k is computed to avoid this, but keep safe.
                    ys = ys[: i + 1]
                    candidates = candidates[: i + 1]
                    break

            # Select best candidate this iteration
            best_i = int(np.argmin(ys))
            cand_x = candidates[best_i]
            cand_y = float(ys[best_i])

            improved = cand_y < best_y

            if improved:
                best_x = cand_x.copy()
                best_y = cand_y
                center = best_x.copy()
                sigma *= shrink_on_improve
            else:
                sigma *= shrink_on_no_improve

            # Occasional reset / exploration to avoid stagnation
            # (Only if sigma has room to grow; bounds may be tight.)
            if sigma > 0 and np.random.rand() < reset_prob:
                sigma = max(sigma, reset_factor * scale)

            # If sigma becomes effectively zero due to contraction and box width,
            # allow minor re-expansion within feasible range.
            if sigma <= 1e-15:
                # If box has nonzero width, set a tiny but nonzero sigma.
                if np.any(nonzero):
                    sigma = 1e-6 * float(np.max(box))
                else:
                    # Bounds are degenerate; no movement possible.
                    break

            if evals >= budget:
                break

        return best_x, best_y
