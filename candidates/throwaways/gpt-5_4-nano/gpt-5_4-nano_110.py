# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization method that combines
# derivative-free coordinate perturbations with an adaptive “step size” and
# occasional restart-like diversification. It is designed to be robust across
# dimensions and only uses numpy.
# Search state: Maintains a current best point x_best and its value f_best,
# plus a working point x_curr and a global step size sigma. Also tracks how
# many evaluations have been used.
# Candidate generation: Each iteration proposes a small batch of trial points
# by moving along randomly chosen coordinate directions (signed), plus a
# few purely random points. Coordinate steps scale with sigma, and the
# random points are uniform within the provided bounds.
# Selection and replacement: Among all candidates, picks the one with the lowest
# objective value. If it improves over f_best, updates x_best, x_curr, and
# increases the “confidence” by slightly expanding sigma; otherwise decreases
# sigma to refine locally.
# Adaptation: sigma is adapted based on success/failure: improvement triggers
# moderate sigma increase (encouraging exploration), while stagnation triggers
# sigma decrease (encouraging exploitation).
# Exploration mechanisms: The algorithm injects random samples with a probability
# that grows when sigma becomes small, helping escape local minima.
# Exploitation mechanisms: Most candidates are constructed by coordinate
# perturbations around the current best, targeting local improvement.
# Boundary handling: Proposed points are clipped to the feasible domain
# [lb, ub] on every evaluation, ensuring feasibility without discarding budget.
# Budget strategy: Calls func(x) exactly up to the provided evaluation budget.
# Each batch uses min(batch_size, remaining_budget) candidates.
# Closest known influences: A light hybrid of coordinate pattern search and
# (1+λ)-ES style success-based step-size control, adapted for bounded
# continuous spaces.
# Novelty or unusual aspects: Uses a coordinate-wise perturbation scheme with
# an adaptive mixture of local and uniform proposals, plus automatic scaling of
# step magnitude relative to the bounds range.
# Failure modes: If the objective is extremely irregular, very flat, or highly
# deceptive, the step-size adaptation may shrink sigma too quickly; diversification
# mitigates this but cannot guarantee success under severe adversarial
# conditions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed: return a feasible point anyway.
            lb, ub = self._get_bounds(func, dim)
            x0 = (lb + ub) / 2.0
            return np.asarray(x0, dtype=float), float("inf")

        lb, ub = self._get_bounds(func, dim)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        # Ensure valid ranges (handle accidental reversed bounds).
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        width = hi - lo
        # Avoid zero-width dimensions (make step 0 there).
        safe_width = np.where(width > 0, width, 1.0)

        rng = np.random

        evals = 0
        best_x = None
        best_y = float("inf")

        def clip(x):
            return np.minimum(np.maximum(x, lo), hi)

        def eval_point(x):
            nonlocal best_x, best_y, evals
            if evals >= budget:
                # Should never happen if budget logic is correct.
                return best_y
            x = clip(x)
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.array(x, dtype=float, copy=True)
            return y

        # Initial point: uniform random within bounds.
        # Using bounds ensures feasibility under clipping.
        x_curr = lo + rng.rand(dim) * safe_width
        best_x = np.array(x_curr, dtype=float, copy=True)
        best_y = float(func(clip(best_x)))
        evals = 1

        # Adaptive step size: proportional to search space.
        # Start modestly to avoid early clipping.
        sigma = 0.25 * np.mean(safe_width)
        sigma = max(sigma, 1e-12)

        # Parameters: local batch size and random restart frequency.
        # Kept small to preserve budget and adapt quickly.
        base_local = 4
        base_random = 1

        # Main loop: each iteration proposes a batch.
        while evals < budget:
            remaining = budget - evals

            # Batch size scales with dimension but stays bounded.
            batch_local = min(base_local + dim // 4, remaining)
            # Allocate a few random points in the batch when sigma is small.
            # If sigma is large, mostly exploit around best.
            # Probability of extra randomness increases as sigma shrinks.
            shrink_frac = 1.0 - min(1.0, sigma / (0.25 * np.mean(safe_width) + 1e-12))
            extra_random = 1 if rng.rand() < (0.15 + 0.5 * shrink_frac) else 0
            batch_random = min(base_random + extra_random, remaining - batch_local)
            if batch_random < 0:
                batch_random = 0

            candidates = []

            # Exploitation: coordinate-wise perturbations around current best.
            # Each candidate perturbs one coordinate by +/- step.
            # Step magnitude varies by coordinate based on range.
            for _ in range(batch_local):
                x = np.array(best_x, copy=True)
                k = int(rng.randint(0, dim))
                sign = -1.0 if rng.rand() < 0.5 else 1.0
                # Coordinate scaling by that dimension's width.
                coord_step = sigma * (safe_width[k] / (np.mean(safe_width) + 1e-12))
                # Add mild gaussian noise to avoid purely axis-aligned traps.
                noise = 0.2 * sigma * rng.randn()
                x[k] = x[k] + sign * coord_step + noise
                candidates.append(x)

            # Exploration: uniform random samples in the domain.
            for _ in range(batch_random):
                x = lo + rng.rand(dim) * safe_width
                candidates.append(x)

            # If budget is very tight, ensure we don't evaluate too many.
            if len(candidates) > remaining:
                candidates = candidates[:remaining]

            # Evaluate candidates and pick best improvement.
            y_before = best_y
            best_candidate_y = best_y
            best_candidate_x = None

            for x in candidates:
                y = float(func(clip(np.array(x, dtype=float, copy=False))))
                # Update best tracking via eval_point to keep consistent count.
                # Since we already called func, emulate eval tracking here.
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = clip(np.array(x, dtype=float, copy=True))
                if y < best_candidate_y:
                    best_candidate_y = y
                    best_candidate_x = clip(np.array(x, dtype=float, copy=True))

                if evals >= budget:
                    break

            # Adapt sigma based on improvement.
            if best_y < y_before:
                # Success: expand slightly to explore wider neighborhood.
                sigma *= 1.15
            else:
                # Failure: shrink to exploit locally.
                sigma *= 0.7

            # Prevent sigma from becoming too small or too large.
            sigma_max = 0.5 * np.mean(safe_width) + 1e-12
            sigma_min = 1e-12
            sigma = min(sigma_max, max(sigma_min, sigma))

            # Optional diversification when sigma is tiny.
            # We do it sparsely to save budget.
            if sigma <= 2e-12 and (evals < budget) and rng.rand() < 0.3:
                # One random jump attempt.
                if evals < budget:
                    x_jump = lo + rng.rand(dim) * safe_width
                    _ = eval_point(x_jump)

        return np.asarray(best_x, dtype=float), float(best_y)

    @staticmethod
    def _get_bounds(func, dim):
        # Priority: func.lower/func.upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            # Support b.lb/b.ub or b.lower/b.upper
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            elif hasattr(b, "lower") and hasattr(b, "upper"):
                lb = np.asarray(b.lower, dtype=float)
                ub = np.asarray(b.upper, dtype=float)
            else:
                raise AttributeError("func.bounds must provide lb/ub or lower/upper.")
        else:
            raise AttributeError(
                "Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )

        if lb.shape[0] != dim or ub.shape[0] != dim:
            # Allow scalar bounds broadcast for convenience.
            if lb.size == 1:
                lb = np.full(dim, float(lb))
            if ub.size == 1:
                ub = np.full(dim, float(ub))

        if lb.shape[0] != dim or ub.shape[0] != dim:
            raise ValueError(f"Bounds dimension mismatch: expected dim={dim}, got lb={lb.shape}, ub={ub.shape}")
        return lb, ub
