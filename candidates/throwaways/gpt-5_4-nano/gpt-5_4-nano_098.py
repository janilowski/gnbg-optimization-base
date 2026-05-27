# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimizer
# (GNBG-style) using a trust-region-like coordinate search with adaptive step
# sizes and occasional random restarts. It works for arbitrary dimension and
# respects a strict function-evaluation budget.
# Search state: The algorithm maintains a current best point x_best/y_best,
# a step size sigma, and a set of candidate points generated from orthogonal
# (coordinate) perturbations around the current best.
# Candidate generation: At each iteration, it samples points by moving along
# coordinate directions (+/-) scaled by sigma, also including a small amount of
# isotropic Gaussian noise to avoid stagnation.
# Selection and replacement: Among evaluated candidates (including the incumbent),
# it selects the best (minimum) value. If improvement is found, the new point
# becomes the incumbent.
# Adaptation: The step size sigma shrinks when no improvement is found and grows
# modestly after improvements to keep exploration effective.
# Exploration mechanisms: If repeated stagnation occurs or sigma becomes too small,
# the algorithm performs a randomized restart near the center of the feasible
# domain (or around the incumbent if bounds are unavailable).
# Exploitation mechanisms: When improvement is observed, the search concentrates
# around the new incumbent by continuing coordinate perturbations with the
# updated sigma.
# Boundary handling: Candidate points are clipped to the provided bounds at
# every evaluation.
# Budget strategy: The total number of objective evaluations is capped at
# "budget". Each evaluation is counted, and candidate batches are sized so
# they never exceed the remaining budget.
# Closest known influences: Combines ideas from coordinate pattern search,
# shrinking trust regions, and restart-based global escape, tuned for small
# black-box budgets.
# Novelty or unusual aspects: Uses a structured coordinate ensemble plus isotropic
# perturbations, with strict budget accounting and dimension-adaptive candidate
# batch sizing.
# Failure modes: If the objective is extremely rugged or the budget is tiny,
# the algorithm may only find a local improvement or fail to escape poor regions.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        if budget <= 0:
            raise ValueError("budget must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._get_bounds(func, self.dim)

        # Handle degenerate bounds (lb == ub) by keeping x fixed there.
        span = ub - lb
        span_safe = np.where(span > 0, span, 1.0)  # for scaling without division by zero

        evals_used = 0

        def evaluate(x):
            nonlocal evals_used
            if evals_used >= self.budget:
                # Should not happen due to careful accounting, but keep safe.
                # Return current best as a fallback.
                return best_y
            x = np.clip(x, lb, ub)
            y = float(func(x))
            evals_used += 1
            return y

        # Initialize incumbent: random point within bounds.
        # Harness sets np seed before each run.
        x0 = lb + np.random.rand(self.dim) * (ub - lb)
        if self.dim == 1:
            x0 = np.array([x0], dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float)

        best_x = x0
        best_y = evaluate(best_x)

        # Initial step size: fraction of domain scale.
        # Use median span to be robust across mixed scales.
        base_scale = np.median(span_safe)
        sigma = 0.25 * base_scale if base_scale > 0 else 0.25

        # Stagnation counter for triggering restart/shrink.
        no_improve_iters = 0
        max_no_improve = 8  # dimension-independent heuristic

        # Main loop: stop based on budget.
        # Each iteration evaluates a batch of candidates but never beyond remaining budget.
        # We use a simple pattern: coordinate +/- plus one random jittered point.
        while evals_used < self.budget:
            remaining = self.budget - evals_used
            if remaining <= 0:
                break

            # Choose number of coordinates to try this iteration.
            # If budget is tight, evaluate fewer coordinates.
            # Candidate count = 2*k (+ maybe jitter) + optionally keep incumbent.
            # We'll always evaluate candidates explicitly; incumbent already evaluated.
            # Batch size calculation keeps within remaining.
            k_max = self.dim
            # We'll allocate up to roughly 1/6 of remaining for the + and - pairs.
            # Ensures at least something gets evaluated.
            # Let n_pairs = k; n_evals = 2*k (+1 for jitter)
            # => 2*k + 1 <= remaining  (if jitter used)
            # Conservative: allow jitter only if enough budget.
            jitter = remaining >= 2 * max(1, min(k_max, self.dim)) + 1

            # Decide k: largest such that 2*k + (1 if jitter) <= remaining
            if jitter:
                k = min(k_max, max(1, (remaining - 1) // 2))
            else:
                k = min(k_max, max(1, remaining // 2))
            k = int(k)

            # Select k coordinates: random subset to reduce bias in high dimensions.
            coords = np.random.choice(self.dim, size=k, replace=False)

            # Build candidate points
            # Coordinate step magnitude: scaled by per-dimension span to keep comparable.
            # For dims with tiny span, step becomes tiny (and clip will make it exact).
            dim_scales = span_safe
            # Create perturbations (+/-) along chosen coordinates
            candidates = []

            for j in coords:
                step = sigma * dim_scales[j]
                x_plus = np.array(best_x, copy=True)
                x_minus = np.array(best_x, copy=True)
                x_plus[j] = x_plus[j] + step
                x_minus[j] = x_minus[j] - step
                candidates.append(x_plus)
                candidates.append(x_minus)

            # Add one isotropic jitter candidate for exploration when possible
            if jitter:
                # Small fraction of sigma; use normal noise.
                noise = np.random.normal(0.0, 0.25, size=self.dim) * sigma * np.sqrt(dim_scales.mean())
                xj = np.array(best_x, copy=True) + noise
                candidates.append(xj)

            # If candidates exceed remaining due to any rounding, truncate.
            if len(candidates) > remaining:
                candidates = candidates[:remaining]

            # Evaluate and select best among candidates
            # (Strict minimization.)
            iter_best_y = best_y
            iter_best_x = best_x
            for x in candidates:
                y = evaluate(x)
                if y < iter_best_y:
                    iter_best_y = y
                    iter_best_x = np.array(x, copy=True)

                # Early break if budget exhausted
                if evals_used >= self.budget:
                    break

            # Update incumbent and adapt sigma
            if iter_best_y < best_y - 1e-15:
                best_y = iter_best_y
                best_x = iter_best_x
                no_improve_iters = 0
                # Increase sigma slightly to exploit progress
                sigma = sigma * 1.15
            else:
                no_improve_iters += 1
                # Shrink sigma when no improvement
                sigma = sigma * 0.65

            # Restart mechanism when stagnating or sigma too small
            if no_improve_iters >= max_no_improve or sigma < 1e-14:
                # Restart near center of the domain with larger step.
                center = 0.5 * (lb + ub)
                # Choose restart scale based on domain span
                domain_scale = np.median(span_safe)
                sigma_restart = 0.5 * domain_scale if domain_scale > 0 else sigma
                if sigma_restart <= 0:
                    sigma_restart = 0.5
                x_restart = center + np.random.normal(0.0, 1.0, size=self.dim) * sigma_restart * 0.5
                y_restart = evaluate(x_restart)
                if y_restart < best_y:
                    best_y = y_restart
                    best_x = np.clip(x_restart, lb, ub)
                # Reset sigma to a moderate value
                sigma = max(0.25 * sigma_restart, sigma)
                no_improve_iters = 0

        return best_x, best_y

    @staticmethod
    def _get_bounds(func, dim: int):
        """
        Read bounds from either:
        - func.lower / func.upper
        - func.bounds.lb / func.bounds.ub
        Returns (lb, ub) as float numpy arrays of shape (dim,).
        """
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
            ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)
        else:
            raise AttributeError(
                "func must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub"
            )

        if lb.size != dim or ub.size != dim:
            # Allow broadcasting if bounds are scalars
            if lb.size == 1 and ub.size == 1:
                lb = np.full(dim, float(lb.item()), dtype=float)
                ub = np.full(dim, float(ub.item()), dtype=float)
            else:
                raise ValueError(f"Bounds must match dim={dim}. Got lb size {lb.size}, ub size {ub.size}.")

        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            raise ValueError("Bounds must be finite.")

        # Ensure lb <= ub (if reversed, swap silently)
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        return lo, hi
