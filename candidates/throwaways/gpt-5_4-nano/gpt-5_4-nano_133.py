# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm inspired by
# CMA-ES-like evolution with coordinate-wise step-size adaptation, but without
# any external dependencies beyond numpy. It maintains a population of candidate
# solutions, samples from a multivariate Gaussian around the current mean,
# evaluates them on the provided objective, and updates the mean and step size
# based on weighted best candidates.
# Search state: A current mean vector `x_mean`, a global step size `sigma`,
# an axis-aligned covariance approximation given by `diag(var)` (kept as a
# variance scale per dimension), and bookkeeping for remaining evaluations.
# Candidate generation: Each iteration samples a population by adding Gaussian
# perturbations to `x_mean`. Perturbations use `sigma * sqrt(var)` per
# coordinate, then candidates are clipped to the provided bounds.
# Selection and replacement: Candidates are sorted by objective value (lower is
# better). The top fraction with nonzero weights are used to update `x_mean`
# via a weighted recombination toward promising points.
# Adaptation: A rank-based update increases/decreases `sigma` based on the
# relative improvement of the best candidate versus the current mean. Variance
# scales are nudged using the spread of successful candidates, with damping
# to keep behavior stable across dimensions.
# Exploration mechanisms: Stochastic sampling with nontrivial sigma and
# variance keeps exploring; clipping to bounds still allows movement inside the
# feasible region.
# Exploitation mechanisms: Selection pressure via weighted recombination moves
# the mean toward high-quality candidates; sigma adaptation reduces step size
# when progress is observed.
# Boundary handling: Candidates are clipped to bounds derived from
# `func.lower/func.upper` or `func.bounds.lb/func.bounds.ub`. The mean is also
# clipped after updates.
# Budget strategy: The algorithm converts the given evaluation `budget` into a
# number of iterations and population size; it never evaluates more than the
# budget by adjusting the final generation size.
# Closest known influences: Rank-based evolutionary strategies / CMA-ES variants
# (mean update with weighted bests; adaptive step size using success signals),
# simplified to an axis-aligned covariance for compactness.
# Novelty or unusual aspects: Uses axis-aligned variance scaling with a
# lightweight, success-based sigma update and variance damping to remain robust
# while staying small and readable. Does not require gradient or function
# internals besides bounds.
# Failure modes: If bounds are extremely tight, clipping can reduce diversity
# and cause premature stagnation. On very noisy or deceptive objectives, the
# success-based adaptation may misinterpret noise as progress; the algorithm
# mitigates this with conservative damping and a minimum sigma.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        if d <= 0:
            raise ValueError("dim must be positive")
        if self.budget <= 0:
            raise ValueError("budget must be positive")

        # ---- Bounds handling (required by prompt) ----
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float).reshape(-1)
            ub = np.asarray(func.bounds.ub, dtype=float).reshape(-1)
        else:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub.")

        if lb.size != d or ub.size != d:
            raise ValueError("Bounds size does not match dim.")

        # Ensure finite and valid ranges (robustness)
        lb = np.where(np.isfinite(lb), lb, -1e3)
        ub = np.where(np.isfinite(ub), ub, 1e3)
        if np.any(ub < lb):
            raise ValueError("Invalid bounds: ub must be >= lb for all dimensions.")

        span = ub - lb
        # Avoid zero spans causing degenerate scaling
        span = np.where(span > 0, span, 1.0)

        # ---- Evaluation budget guard ----
        evals_used = 0
        budget = self.budget

        def eval_one(x):
            nonlocal evals_used
            if evals_used >= budget:
                # Should not happen; but keep hard guard.
                raise RuntimeError("Evaluation budget exceeded.")
            y = func(np.asarray(x, dtype=float))
            evals_used += 1
            return float(np.asarray(y).reshape(()))

        # ---- Initialize mean randomly within bounds ----
        rng = np.random
        x_mean = lb + rng.rand(d) * (ub - lb)
        x_mean = np.clip(x_mean, lb, ub)

        # Initial sigma: fraction of typical span
        sigma = 0.3 * float(np.median(span))
        sigma = max(sigma, 1e-12)

        # Axis-aligned variance scaling (starts uniform)
        var = np.ones(d, dtype=float)

        # Population sizing: choose modest size to fit budget across dimensions
        # Ensure at least 2 and at most 32 (but also not exceeding budget).
        pop = int(np.clip(4 + d // 2, 2, 32))
        pop = min(pop, budget)  # can't evaluate more than budget

        # Determine number of full generations we can afford.
        # Each generation evaluates `gen_pop` candidates.
        # We will also evaluate mean once as a baseline.
        eval_baseline = 1 if budget >= 1 else 0
        best_x = np.array(x_mean, copy=True)
        best_y = eval_one(x_mean) if eval_baseline else np.inf

        # We will always start from baseline mean; remaining budget for iterations
        remaining = budget - evals_used
        if remaining <= 0:
            return best_x, best_y

        # Choose number of generations (at least 1)
        # Each generation uses up to `pop` evals.
        gens = max(1, remaining // pop)
        # It's okay if last generation underuses budget; we adjust.

        # ---- Evolution strategy loop ----
        # Weighted recombination: top-k weights sum to 1
        # Common pattern: weights proportional to log.
        k = max(2, pop // 2)
        k = min(k, pop)
        ranks = np.arange(k)
        # log-based positive weights
        w = np.log((k + 1) / 2.0) - np.log(k + 1 + ranks)
        w = np.maximum(w, 0.0)
        if w.sum() == 0:
            w = np.ones_like(w)
        w = w / w.sum()

        # sigma adaptation parameters
        c_sigma = 0.3  # learning rate for sigma update
        d_sigma = 1.0   # speed
        # Conservative damping for var updates
        c_var = 0.2
        var_floor = 1e-12

        # Success signal: compare best in generation against best_y baseline
        for _ in range(gens):
            remaining = budget - evals_used
            if remaining <= 0:
                break

            gen_pop = min(pop, remaining)
            # Candidate generation
            # Sample N(0, I) then scale by diag(sqrt(var)) and sigma.
            # (Axis-aligned covariance approximation for compactness.)
            Z = rng.randn(gen_pop, d)
            step = (sigma * np.sqrt(var))[None, :] * Z
            X = x_mean[None, :] + step

            # Boundary handling: clip to bounds (feasible candidates)
            X = np.clip(X, lb, ub)

            # Evaluate
            Y = np.empty(gen_pop, dtype=float)
            for i in range(gen_pop):
                Y[i] = eval_one(X[i])

            # Keep global best
            idx_best = int(np.argmin(Y))
            if Y[idx_best] < best_y:
                best_y = float(Y[idx_best])
                best_x = np.array(X[idx_best], copy=True)

            # Sort candidates for selection
            order = np.argsort(Y)
            X_sorted = X[order]
            Y_sorted = Y[order]

            # Weighted recombination toward top-k
            take = min(k, gen_pop)
            X_top = X_sorted[:take]
            # If take < k, recompute weights proportionally
            if take != k:
                ranks_t = np.arange(take)
                w_t = np.log((take + 1) / 2.0) - np.log(take + 1 + ranks_t)
                w_t = np.maximum(w_t, 0.0)
                if w_t.sum() == 0:
                    w_t = np.ones_like(w_t)
                w_t = w_t / w_t.sum()
                x_new = np.dot(w_t, X_top)
            else:
                x_new = np.dot(w, X_top)

            # Success-based sigma update:
            # If best improves sufficiently, shrink less / increase exploration slightly,
            # otherwise shrink sigma (more exploitation / reduce random walk).
            best_gen = float(Y_sorted[0])
            # Normalize improvement relative to magnitude; avoid division by zero
            denom = abs(best_y) + 1.0
            improvement = (best_y - best_gen) / denom  # positive if best_gen < best_y
            # Use a bounded nonlinearity
            # improvement > 0 => sigma tends to increase; otherwise decrease
            # This is rank/relative, robust to scale.
            success = np.tanh(3.0 * improvement)
            sigma_factor = np.exp(c_sigma * (success / d_sigma))
            sigma *= sigma_factor

            # Clip sigma to reasonable bounds based on span
            sigma_min = 1e-12 * float(np.median(span))
            sigma_max = 2.0 * float(np.max(span))
            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # Variance (axis) adaptation:
            # Use deviations of top candidates around current mean.
            # Update var with damping to remain stable.
            # Compute normalized squared step directions.
            # If sigma is tiny, var update is still stable due to floor.
            spread = X_top - x_mean[None, :]
            # Weighted average squared deviations per coordinate
            if take != k:
                # recompute weights for take
                ranks_t = np.arange(take)
                w_t = np.log((take + 1) / 2.0) - np.log(take + 1 + ranks_t)
                w_t = np.maximum(w_t, 0.0)
                if w_t.sum() == 0:
                    w_t = np.ones_like(w_t)
                w_t = w_t / w_t.sum()
                sq = (spread ** 2) * w_t[:, None]
            else:
                sq = (spread ** 2) * w[:take, None]
            new_var = np.maximum(sq.sum(axis=0) / (sigma ** 2 + 1e-30), var_floor)

            # Smooth update of var
            var = (1.0 - c_var) * var + c_var * new_var
            var = np.maximum(var, var_floor)

            # Replacement: move mean
            x_mean = np.clip(np.asarray(x_new, dtype=float), lb, ub)

            # Early exit if budget tight (but we still respect budget guard)
            if evals_used >= budget:
                break

        return np.asarray(best_x, dtype=float), float(best_y)
