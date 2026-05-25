# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, budget-aware black-box minimization
# algorithm for continuous domains. It is a hybrid of coordinate-wise random search
# and a Gaussian-like local refinement around the current best point.
# Search state: The algorithm maintains an incumbent best solution (x_best, y_best),
# a step size sigma controlling perturbations, and a small set of candidate evaluations
# per iteration.
# Candidate generation: Each iteration generates candidates by perturbing x_best with
# (1) random isotropic Gaussian steps and (2) sparse coordinate steps. Additionally,
# it samples a few purely random points early on to avoid poor initializations.
# Selection and replacement: Among evaluated candidates, the algorithm updates the incumbent
# if any candidate improves the best observed value.
# Adaptation: The step size sigma is adapted using a simple 1/5-success rule based on whether
# recent candidates improved the incumbent.
# Exploration mechanisms: Early-stage random sampling and occasional large perturbations
# help exploration.
# Exploitation mechanisms: Later iterations focus on local perturbations around x_best with
# decreasing sigma.
# Boundary handling: Candidates are clipped to the provided bounds after each perturbation.
# Budget strategy: The total number of objective calls is capped by `budget`. The code tracks
# remaining evaluations and chooses how many candidates to evaluate each iteration to never
# exceed the budget. No additional (wasted) evaluations are performed after budget exhaustion.
# Closest known influences: The design is loosely inspired by evolution strategies / CMA-ES-style
# local search, combined with coordinate perturbations and an adaptive step size.
# Novelty or unusual aspects: Uses sparse coordinate perturbations in addition to isotropic
# steps, and adapts sigma based on improvements within each batch under a strict global
# evaluation budget.
# Failure modes: If the objective is extremely noisy or has very narrow feasible optima,
# clipping and Gaussian perturbations may waste evaluations; performance may degrade.
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
            # No evaluations allowed: return a valid point (midpoint) but unknown objective.
            x = self._get_bounds_center(func, dim)
            return x, float("inf")

        lb, ub = self._get_bounds(func, dim)
        lb = lb.astype(float)
        ub = ub.astype(float)

        # If bounds are degenerate, sigma becomes 0; still must run robustly.
        span = ub - lb
        span[span == 0.0] = 0.0

        # Initialize best with one evaluation at a reasonable start.
        # Start point: midpoint with small random jitter to break symmetry.
        x_best = self._clip(
            self._rand_uniform(lb, ub) if np.all(span > 0) else self._get_bounds_center(func, dim),
            lb,
            ub,
        )
        y_best = self._eval(func, x_best, budget_left=budget)
        budget_left = budget - 1
        if budget_left <= 0:
            return x_best, y_best

        # Initial sigma: fraction of domain span, falling back to a small scale.
        # Use geometric mean of span to remain dimension-agnostic.
        positive_span = span[span > 0]
        if positive_span.size:
            gm = float(np.exp(np.mean(np.log(positive_span))))
            sigma = 0.25 * gm
        else:
            sigma = 0.1

        # Batch size: keep small for overhead and to adapt quickly.
        # Ensure at least 1 evaluation per loop.
        base_batch = 1 + min(6, max(1, dim // 2))
        # 1/5 success rule parameters
        target_success = 0.2
        adapt_up = 1.5
        adapt_down = 0.7

        # Early exploration: draw a few global points.
        # Number of early random points.
        early_k = min(3 * base_batch, max(0, budget_left // 4))
        success_counter = 0
        attempts_counter = 0

        # Helper: evaluate candidates in a loop without exceeding budget.
        def eval_batch(candidates):
            nonlocal y_best, x_best, budget_left, success_counter, attempts_counter
            best_local_improved = False
            for x in candidates:
                if budget_left <= 0:
                    return False
                y = func(x)
                budget_left -= 1
                attempts_counter += 1
                if y < y_best:
                    y_best = float(y)
                    x_best = x.copy()
                    success_counter += 1
                    best_local_improved = True
                if budget_left <= 0:
                    return best_local_improved
            return best_local_improved

        # Early random exploration
        if early_k > 0 and budget_left > 0:
            k = min(early_k, budget_left)
            # Sample uniformly within bounds.
            candidates = self._rand_uniform(lb, ub, size=(k, dim))
            eval_batch(candidates)

        # Main loop
        # Each iteration evaluates up to `batch` candidates, adapting sigma after.
        while budget_left > 0:
            # Determine batch size dynamically.
            batch = min(base_batch, budget_left)
            if batch <= 0:
                break

            # Mix of isotropic Gaussian steps and sparse coordinate moves.
            # Increase exploitation as sigma shrinks automatically.
            candidates = np.empty((batch, dim), dtype=float)

            # Use sparse moves for some candidates, isotropic for the rest.
            # Roughly: 40% coordinate, 60% isotropic.
            n_coord = int(round(0.4 * batch))
            n_iso = batch - n_coord

            # Isotropic candidates
            if n_iso > 0:
                noise = np.random.randn(n_iso, dim)
                # Scale noise: sigma relative to span magnitude to be robust across dimensions.
                # Normalize by sqrt(dim) to keep perturbation magnitude stable.
                scale = sigma / max(1.0, np.sqrt(dim))
                candidates[:n_iso] = x_best + scale * noise

            # Coordinate candidates: choose one coordinate per candidate.
            if n_coord > 0:
                # For each candidate, pick a coordinate and perturb only it.
                idx = np.random.randint(0, dim, size=n_coord)
                # Perturbation magnitudes: slightly larger than isotropic typical.
                # Use independent Gaussian for step sign and magnitude.
                coord_noise = np.random.randn(n_coord) * sigma
                cand = np.tile(x_best, (n_coord, 1))
                cand[np.arange(n_coord), idx] = cand[np.arange(n_coord), idx] + coord_noise
                candidates[n_iso:] = cand

            # Occasionally include a "large" exploratory step when sigma isn't too small.
            # This helps avoid stagnation. Use one candidate if batch allows.
            if batch >= 2 and budget_left > 0 and sigma > 1e-12:
                if np.random.rand() < 0.15:
                    # Replace last candidate with a larger perturbation.
                    big_noise = np.random.randn(dim)
                    candidates[-1] = x_best + (2.0 * sigma) * big_noise / max(1.0, np.sqrt(dim))

            # Clip to bounds
            for i in range(batch):
                candidates[i] = self._clip(candidates[i], lb, ub)

            # Track successes for adaptation
            prev_success = success_counter
            prev_attempts = attempts_counter
            eval_batch(candidates)

            # Adapt sigma based on success rate in this batch
            # Count only new attempts since previous adaptation.
            batch_attempts = attempts_counter - prev_attempts
            batch_success = success_counter - prev_success
            if batch_attempts > 0:
                success_rate = batch_success / batch_attempts
                # 1/5 rule: increase if too successful, decrease if not.
                if success_rate > target_success:
                    sigma *= adapt_up
                else:
                    sigma *= adapt_down

            # Avoid sigma collapsing to 0 in degenerate spaces
            # If span is non-zero, keep a minimal fraction for robustness.
            if np.any(span > 0):
                min_sigma = 1e-12 * float(np.max(span))
                sigma = max(sigma, min_sigma)

        return x_best, y_best

    @staticmethod
    def _clip(x, lb, ub):
        # Works for 1d arrays and scalars.
        return np.minimum(np.maximum(x, lb), ub)

    @staticmethod
    def _rand_uniform(lb, ub, size=None):
        # Uniform sampling in [lb, ub], handling vectors.
        # If size is None: return shape (dim,)
        if size is None:
            r = np.random.rand(lb.shape[0])
        else:
            r = np.random.rand(size[0], lb.shape[0])
        return lb + (ub - lb) * r

    @staticmethod
    def _get_bounds_center(func, dim):
        lb, ub = Algorithm._get_bounds(func, dim)
        lb = lb.astype(float)
        ub = ub.astype(float)
        return 0.5 * (lb + ub)

    @staticmethod
    def _get_bounds(func, dim):
        # Read bounds from either func.lower/func.upper or func.bounds.lb/func.bounds.ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            lb = np.asarray(getattr(b, "lb"), dtype=float)
            ub = np.asarray(getattr(b, "ub"), dtype=float)
        else:
            # If bounds are missing, default to [-1, 1] for robustness.
            lb = np.full(dim, -1.0, dtype=float)
            ub = np.full(dim, 1.0, dtype=float)

        if lb.shape[0] != dim:
            # Broadcast scalar-like or incorrect shape
            if lb.size == 1:
                lb = np.full(dim, float(lb), dtype=float)
            else:
                lb = lb.reshape(-1)
                if lb.size >= dim:
                    lb = lb[:dim]
                else:
                    lb = np.pad(lb, (0, dim - lb.size), mode="edge")

        if ub.shape[0] != dim:
            if ub.size == 1:
                ub = np.full(dim, float(ub), dtype=float)
            else:
                ub = ub.reshape(-1)
                if ub.size >= dim:
                    ub = ub[:dim]
                else:
                    ub = np.pad(ub, (0, dim - ub.size), mode="edge")

        # Ensure lb <= ub
        swap = lb > ub
        if np.any(swap):
            tmp = lb.copy()
            lb[swap] = ub[swap]
            ub[swap] = tmp[swap]

        return lb, ub

    @staticmethod
    def _eval(func, x, budget_left):
        # Single evaluation; kept separate to make budget intent explicit.
        # Assumes budget_left > 0 is already checked by caller.
        y = func(x)
        return float(y)
