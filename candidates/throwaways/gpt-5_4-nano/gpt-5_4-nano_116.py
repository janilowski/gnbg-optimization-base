# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# based on evolutionary search with a CMA-ES–like diagonal adaptation and a
# restart mechanism. It repeatedly samples candidate points from a Gaussian
# distribution, evaluates the objective, keeps the best solutions, and updates
# sampling step-sizes using rank-based information.
# Search state: The algorithm maintains a mean vector (current center), a
# vector of per-dimension step sizes (sigma), an evaluation counter, and
# a restart counter. Each iteration samples a small population around the mean.
# Candidate generation: At each iteration, it samples lambda points using
# x = clip(mean + sigma * N(0,1), lower, upper). It also evaluates the
# current mean once per iteration to improve stability.
# Selection and replacement: After evaluating the population, it selects the best
# mu individuals by fitness (lowest objective value). The mean is moved toward
# their weighted average.
# Adaptation: Step sizes (sigma) are adapted using improvement-based feedback:
# if the best value improved sufficiently, sigma is shrunk/expanded based on a
# simple success rule and a rank-based measure. This keeps the search stable
# without requiring full covariance estimation.
# Exploration mechanisms: The Gaussian sampling and occasional restarts provide
# global exploration. When progress stalls, sigma is increased and/or the
# mean is reinitialized around a random point within bounds.
# Exploitation mechanisms: The mean update toward top-ranked samples focuses
# search locally, while shrinking sigma promotes exploitation.
# Boundary handling: All candidates are clipped into the provided bounds. If a
# dimension is degenerate (upper==lower), sigma becomes zero for that dimension.
# Budget strategy: The total number of function evaluations never exceeds the
# provided budget. The algorithm uses an evaluation budget tracker and stops
# early when it would exceed the limit.
# Closest known influences: The design is inspired by CMA-ES-style mean update and
# step-size adaptation, but simplified to diagonal sigma and lightweight restart
# logic for robustness and compactness.
# Novelty or unusual aspects: It uses a deterministic per-iteration evaluation
# cap computed from remaining budget, and a failure/stagnation counter that
# triggers restart or sigma growth.
# Failure modes: On highly constrained landscapes or pathological bounds,
# clipping can bias the search; also, with very small budgets, only a few
# iterations occur which may limit performance.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        lb, ub = self._read_bounds(func, dim)

        # Handle degenerate bounds early
        span = ub - lb
        feasible = np.isfinite(lb) & np.isfinite(ub)
        if not np.all(feasible):
            raise ValueError("Bounds must be finite numbers.")
        if np.any(span < 0):
            raise ValueError("Invalid bounds: require lower <= upper for all dimensions.")

        # If budget is tiny, just evaluate a couple of points safely.
        max_evals = max(0, self.budget)

        # Evaluation wrapper with budget enforcement
        evals = 0

        def eval_obj(x):
            nonlocal evals
            if evals >= max_evals:
                # Should never happen due to guards, but keep robust.
                return np.inf
            y = func(x)
            evals += 1
            return float(y)

        # Choose initial mean as random point inside bounds.
        # Harness sets numpy seed for reproducibility.
        if max_evals == 0:
            # No evaluations allowed; return a valid point but unknown objective.
            x0 = lb.copy()
            x0 += 0.5 * span
            # If bounds allow, random point:
            if np.all(span > 0):
                x0 = lb + np.random.rand(dim) * span
            best_x = x0
            best_y = np.inf
            return best_x, best_y

        if np.all(span == 0):
            best_x = lb.copy()
            best_y = eval_obj(best_x)
            return best_x, best_y

        # Initial sigma: fraction of span, capped to avoid NaNs
        sigma0 = 0.3 * span
        sigma0 = np.where(span == 0, 0.0, sigma0)
        # Avoid exact zeros for non-degenerate dims
        sigma0 = np.where((span != 0) & (sigma0 == 0), 0.1 * span, sigma0)

        mean = lb + 0.5 * span
        # If span nonzero, randomize a bit for symmetry breaking.
        if np.any(span > 0):
            mean = lb + np.random.rand(dim) * span

        # Population sizes: keep small for compactness and budget fit.
        # We'll adapt lambda based on remaining budget and dimension.
        # Typical CMA-ES uses lambda ~ 4 + 3*log(n), mu ~ lambda/2.
        lambda_base = int(np.clip(4 + 3 * np.log(dim + 1.0), 4, 16))
        stagnation = 0
        best_y = np.inf
        best_x = mean.copy()

        # Track recent best improvements
        prev_best = np.inf

        # Restart parameters
        max_restarts = 2 + int(np.log2(dim + 1.0))
        restarts = 0
        # Success-based adaptation
        success = 0
        failure = 0

        # Precompute weighting for mu selection (rank-based, positive decreasing)
        # We'll rebuild weights when lambda changes; keep simple with mu=ceil(lambda/2).
        # Main loop: iterate until evaluation budget is exhausted.
        while evals < max_evals:
            remaining = max_evals - evals

            # Determine lambda/mu for this iteration within remaining budget
            # Evaluate mean once per iteration plus lambda candidates => 1+lambda <= remaining.
            lambda_now = min(lambda_base, max(1, remaining - 1))
            if lambda_now <= 0:
                break
            # mu is how many elites to use; at most lambda_now
            mu = (lambda_now + 1) // 2
            mu = max(1, min(mu, lambda_now))

            # Rank-based weights: w_i proportional to log(mu+1) - log(i)
            # for i in 1..mu (best is i=1). Normalize to sum to 1.
            idx = np.arange(1, mu + 1, dtype=float)
            w = np.log(mu + 1.0) - np.log(idx)
            w = np.maximum(0.0, w)
            if np.sum(w) <= 0:
                w = np.ones(mu, dtype=float)
            w /= np.sum(w)

            # Ensure sigma respects degeneracies
            sigma = np.where(span == 0, 0.0, sigma0)

            # Evaluate current mean (stabilizes selection and handles clipping bias)
            y_mean = eval_obj(mean)
            if y_mean < best_y:
                best_y = y_mean
                best_x = mean.copy()

            # Sample candidates
            # Gaussian sampling around current mean with diagonal sigma
            # Candidate i: mean + sigma * N(0,1), then clip to bounds.
            Z = np.random.randn(lambda_now, dim)
            X = mean[None, :] + (sigma[None, :] * Z)
            X = np.clip(X, lb[None, :], ub[None, :])

            # Evaluate candidates
            Ys = np.empty(lambda_now, dtype=float)
            for i in range(lambda_now):
                Ys[i] = eval_obj(X[i])

            # Select elites (lowest objective)
            order = np.argsort(Ys)
            elite_idx = order[:mu]
            elites = X[elite_idx]
            elite_y = Ys[elite_idx]

            # Update global best
            if elite_y[0] < best_y:
                best_y = float(elite_y[0])
                best_x = elites[0].copy()

            # Mean update: move toward weighted elite average
            new_mean = np.sum(elites * w[:, None], axis=0)
            # If clipping causes no change, still allow update
            mean = new_mean

            # Step-size adaptation (lightweight success rule + rank signal)
            # Compute "improvement" relative to previous best (global best).
            improved = (best_y < prev_best - 1e-12) if np.isfinite(prev_best) else True
            prev_best = best_y

            # Rank signal: how concentrated elites are around the best rank
            # Lower elite_y indicates improvement; map to a success-like scalar.
            # Use normalized difference within elites for robustness.
            elite_best = float(elite_y[0])
            elite_worst = float(elite_y[-1])
            denom = abs(elite_worst) + 1e-12
            progress = (elite_worst - elite_best) / denom  # higher => better separation/improvement

            if improved or progress > 1e-3:
                success += 1
                failure = 0
            else:
                failure += 1
                success = 0

            # Adjust sigma0: shrink on success (exploitation), grow on failure (exploration)
            # Scale factor based on dimension to keep behavior stable.
            if success >= 2:
                # shrink more
                shrink = 0.85 - 0.15 * np.tanh(progress * 10.0)
                sigma0 = np.where(span == 0, 0.0, sigma0 * shrink)
            elif failure >= 2:
                # grow slightly to escape local minima
                grow = 1.15 + 0.2 * np.tanh(progress * 5.0)
                sigma0 = np.where(span == 0, 0.0, sigma0 * grow)

            # Bound sigma0 to reasonable range
            # min sigma is tiny fraction of span; max is full span
            sigma_min = np.where(span == 0, 0.0, 1e-12 * span + 1e-12)
            sigma_max = np.where(span == 0, 0.0, 1.0 * span)
            sigma0 = np.clip(sigma0, sigma_min, sigma_max)

            # Stagnation tracking
            # If global best doesn't improve over several iterations, consider restart.
            # This uses failure as a proxy as well.
            if improved:
                stagnation = 0
            else:
                stagnation += 1

            # Restart when stagnating too long or sigma too small
            # Restart: reinitialize mean randomly and increase sigma.
            too_small = np.all((sigma0 <= 1e-10 * (span + 1e-12)) | (span == 0))
            if (stagnation >= 6 and failure >= 2) or too_small:
                if restarts < max_restarts and evals < max_evals:
                    restarts += 1
                    stagnation = 0
                    failure = 0
                    success = 0
                    # Reinitialize mean; choose random point within bounds
                    if np.any(span > 0):
                        mean = lb + np.random.rand(dim) * span
                        # Inflate sigma to encourage exploration after restart
                        sigma0 = np.where(span == 0, 0.0, 0.5 * span)
                    else:
                        mean = lb.copy()
                        sigma0 = np.zeros(dim, dtype=float)
                else:
                    # If we can't restart, just keep going with expanded sigma a bit
                    sigma0 = np.where(span == 0, 0.0, sigma0 * 1.2)

        return best_x, best_y

    @staticmethod
    def _read_bounds(func, dim):
        # Read bounds from func.lower/upper or func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float).reshape(-1)
            ub = np.asarray(func.upper, dtype=float).reshape(-1)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float).reshape(-1)
                ub = np.asarray(b.ub, dtype=float).reshape(-1)
            else:
                raise AttributeError("func.bounds must have lb and ub attributes.")
        else:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub.")

        if lb.size != dim or ub.size != dim:
            raise ValueError(f"Bounds dimension mismatch: expected dim={dim}, got lb={lb.size}, ub={ub.size}")
        return lb, ub
