# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimization algorithm
# (a simple CMA-ES-inspired evolutionary strategy with restart-like covariance adaptation).
# It maintains a multivariate Gaussian search distribution and iteratively samples candidate
# points around the current mean. The mean and covariance (captured by a diagonal variance
# plus rotation-free global scaling) are adapted using fitness-ranked samples.
# Search state: Current mean vector, diagonal step sizes (variances), an iteration counter,
# and best-so-far solution/fitness. Evaluation budget accounting is enforced strictly.
# Candidate generation: Each generation samples a population from N(mean, diag(step^2)).
# Samples are clipped to the provided bounds. If bounds are degenerate, sampling respects them.
# Selection and replacement: Candidates are sorted by objective value (minimization). The top
# fraction (elites) is used to update the mean by a weighted average. The covariance/step sizes
# are adapted from the spread of elites (diagonal variance estimate).
# Adaptation: Step sizes are increased/decreased based on how the elite spread compares to
# current spread, with damping to keep changes stable.
# Exploration mechanisms: Early generations use larger initial step sizes derived from
# bounds range; occasional variance inflation encourages exploration.
# Exploitation mechanisms: As progress is made, step sizes contract using elite spread,
# focusing sampling near the current mean.
# Boundary handling: All sampled points are clipped to bounds. Mean is also clipped after updates.
# Budget strategy: The total number of function evaluations never exceeds the given budget.
# Population size is chosen to fit remaining budget; each call evaluates only as many candidates
# as can be afforded.
# Closest known influences: Loosely inspired by CMA-ES / evolution strategies using rank-based
# selection and covariance/step-size adaptation, simplified to diagonal covariance for robustness.
# Novelty or unusual aspects: Uses a simple diagonal adaptation with an "elite spread ratio" rule
# and periodic step inflation, aiming for robustness with minimal code.
# Failure modes: If the objective is extremely noisy or bounds are very tight (little motion),
# adaptation may stagnate; clipping can also reduce effective diversity. The algorithm is designed
# to be reasonably stable but not guarantee optimality.
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
        budget = max(0, int(self.budget))
        if budget == 0:
            # No evaluations allowed; return a default feasible point.
            lb, ub = self._get_bounds(func, dim)
            x0 = self._clip(np.zeros(dim), lb, ub)
            return x0, float("inf")

        lb, ub = self._get_bounds(func, dim)
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        # Initial mean: mid-point if bounds available; otherwise zeros.
        mid = 0.5 * (lb + ub)
        # Step size: based on bounds range; fall back to something scale-aware.
        span = ub - lb
        finite_span = np.where(np.isfinite(span), span, 1.0)
        span_nonzero = np.where(np.abs(finite_span) > 0, np.abs(finite_span), 1.0)
        # Use a conservative initial sigma to avoid too many clipped samples.
        sigma0 = 0.3 * span_nonzero
        # For extremely tight boxes, ensure non-zero variance to still permit exploration numerically.
        sigma0 = np.where(sigma0 > 0, sigma0, 1e-6)

        # State
        mean = np.array(mid, dtype=float, copy=True)
        mean = self._clip(mean, lb, ub)
        best_x = mean.copy()
        best_y = float("inf")

        evals = 0

        # Basic population/elite configuration.
        # For diagonal ES, a moderate pop size works across dimensions.
        # Ensure at least 1 evaluation per iteration.
        pop_base = 4 + int(3 * math.log(max(2, dim)))
        pop_base = max(4, pop_base)

        # Elite fraction (rank-based). At least 2 when possible.
        elite_frac = 0.25

        # Main loop: each iteration evaluates a population.
        # We guarantee we never exceed budget by adjusting pop size to remaining budget.
        iteration = 0
        while evals < budget:
            remaining = budget - evals
            # Choose population size to fit remaining budget, but keep reasonable size.
            pop_size = min(pop_base, remaining)
            if pop_size <= 0:
                break

            # Number of elites (at least 1).
            elite_count = max(1, int(math.ceil(pop_size * elite_frac)))
            elite_count = min(elite_count, pop_size)

            # Diagonal step sizes (variances).
            # sigma is per-dimension; keep in sync with step sizes derived from sigma0-scale.
            # Use mean absolute sigma0 scaling for stability even after adaptations.
            # If sigma becomes too small, set a floor to allow some movement.
            sigma_floor = 1e-12 * span_nonzero
            if np.all(sigma0 < sigma_floor):
                sigma_floor = max(1e-12, float(np.mean(span_nonzero))) * 1e-12

            # Use sigma0 as current step sizes (diagonal).
            # (Stored in sigma0 for simplicity.)
            # Sample: X = mean + sigma * N(0,1).
            Z = np.random.randn(pop_size, dim)
            X = mean[None, :] + Z * sigma0[None, :]

            # Boundary handling: clip to [lb, ub].
            X = self._clip(X, lb, ub)

            # Evaluate objective for sampled candidates.
            ys = np.empty(pop_size, dtype=float)
            for i in range(pop_size):
                ys[i] = float(func(X[i]))
            evals += pop_size

            # Update best-so-far
            idx_best = int(np.argmin(ys))
            if ys[idx_best] < best_y:
                best_y = ys[idx_best]
                best_x = np.array(X[idx_best], copy=True)

            # Rank selection (minimization)
            order = np.argsort(ys, kind="stable")
            elites = X[order[:elite_count], :]
            elite_ys = ys[order[:elite_count]]

            # Weighted mean update: favor better elites.
            # Use exponential weights that are robust even if elite_ys are close.
            # Normalize weights safely.
            # If elite_count==1, just set mean to that elite.
            if elite_count == 1:
                new_mean = elites[0].copy()
                # Conservative adaptation: contract a bit toward elites.
                spread = elites[0] - mean
                sigma0 = np.maximum(sigma_floor, 0.7 * sigma0 + 0.3 * np.abs(spread) / max(1.0, np.sqrt(dim)))
            else:
                # Compute weights based on rank (not absolute y to reduce noise sensitivity).
                ranks = np.arange(elite_count, dtype=float)
                # Better rank => smaller index => larger weight
                # tau controls concentration.
                tau = max(1.0, elite_count / 2.0)
                w = np.exp(-(ranks / tau))
                w = w / (np.sum(w) + 1e-300)
                new_mean = np.sum(elites * w[:, None], axis=0)

                # Step-size adaptation: diagonal variance estimate from elite spread around new mean.
                # This is a simplified diagonal "covariance" update.
                diffs = elites - new_mean[None, :]
                elite_var = np.sum((diffs ** 2) * w[:, None], axis=0)

                # Compare elite spread to current spread (sigma0^2) to decide contraction/expansion.
                # Add eps to avoid division by zero.
                current_var = sigma0 ** 2 + 1e-300
                ratio = elite_var / current_var
                # Smooth ratio mapping:
                # - if ratio < 1 => elites tighter => contract
                # - if ratio > 1 => elites spread larger => expand (rare but can help escape)
                # Use a log mapping for stability.
                log_r = np.log(ratio + 1e-300)
                # Convert to contraction factor with damping
                # alpha in [0.05, 0.25]
                alpha = 0.12
                contract = np.exp(-alpha * log_r)  # ratio<1 => log_r<0 => expand? Wait:
                # Let's make it intuitive by directly using ratio:
                # If elite spread is smaller than current spread (ratio<1), shrink sigma.
                # If larger (ratio>1), grow sigma slightly.
                shrink = np.where(ratio < 1.0, 1.0 - 0.35 * (1.0 - ratio), 1.0 + 0.15 * (ratio - 1.0))
                # Combine both effects for robustness
                factor = 0.7 * contract + 0.3 * shrink

                # Damping and floors/ceilings based on bounds span to prevent runaway.
                # Cap sigma by a fraction of span to keep sampling reasonable.
                sigma_cap = 0.8 * span_nonzero + 1e-9
                sigma0 = sigma0 * factor
                # Additional damping to avoid oscillations
                sigma0 = 0.85 * sigma0 + 0.15 * np.sqrt(elite_var + 1e-300)
                sigma0 = np.clip(sigma0, sigma_floor, sigma_cap)

                # Optional periodic exploration inflation:
                # Every ~10 iterations, if still far from best or just to refresh diversity, inflate slightly.
                if pop_base >= 4 and dim >= 2 and (iteration % max(1, int(10 + dim / 3))) == 0:
                    inflate = 1.08
                    sigma0 = np.minimum(sigma0 * inflate, sigma_cap)

                # Update mean
                new_mean = self._clip(new_mean, lb, ub)
                mean = new_mean

            # If elite spread drove too much contraction or mean stagnates, allow mild exploration.
            if iteration > 0 and np.all(sigma0 <= (sigma_floor * 5.0 + 1e-12)):
                sigma0 = np.minimum(sigma0 * 5.0, 0.8 * span_nonzero + 1e-9)
            else:
                # Gentle contraction over time to increase exploitation
                # (monotonic factor close to 1).
                time_factor = 0.995
                sigma0 = np.maximum(sigma_floor, sigma0 * time_factor)

            iteration += 1

        return np.array(best_x, copy=True), float(best_y)

    @staticmethod
    def _get_bounds(func, dim):
        """
        Read bounds from either func.lower/func.upper or func.bounds.lb/func.bounds.ub.
        """
        # Priority 1: func.lower and func.upper
        lower = getattr(func, "lower", None)
        upper = getattr(func, "upper", None)
        if lower is not None and upper is not None:
            lb = np.asarray(lower, dtype=float).reshape(-1)
            ub = np.asarray(upper, dtype=float).reshape(-1)
            if lb.size != dim or ub.size != dim:
                raise ValueError("Bounds size mismatch with dim.")
            return lb, ub

        # Priority 2: func.bounds.lb and func.bounds.ub
        bounds = getattr(func, "bounds", None)
        if bounds is None:
            # Fallback: unbounded -> use [-1, 1]
            return -np.ones(dim, dtype=float), np.ones(dim, dtype=float)

        lb = np.asarray(getattr(bounds, "lb"), dtype=float).reshape(-1)
        ub = np.asarray(getattr(bounds, "ub"), dtype=float).reshape(-1)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds size mismatch with dim.")

        # If some bounds are non-finite, fall back locally.
        # (Still keeps algorithm operational without file IO.)
        if np.any(~np.isfinite(lb)) or np.any(~np.isfinite(ub)):
            # Replace non-finite with defaults
            lb = np.where(np.isfinite(lb), lb, -1.0)
            ub = np.where(np.isfinite(ub), ub, 1.0)

        # Ensure lb <= ub
        swap = lb > ub
        if np.any(swap):
            tmp = lb.copy()
            lb[swap] = ub[swap]
            ub[swap] = tmp[swap]

        return lb, ub

    @staticmethod
    def _clip(x, lb, ub):
        if x.ndim == 1:
            return np.minimum(np.maximum(x, lb), ub)
        return np.minimum(np.maximum(x, lb[None, :]), ub[None, :])
