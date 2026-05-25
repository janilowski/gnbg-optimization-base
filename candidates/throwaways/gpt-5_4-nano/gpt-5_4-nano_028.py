# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization strategy
# (a derivative-free evolution strategy variant with restart-like behavior).
# It maintains a population of candidate solutions, samples new points by adding
# Gaussian perturbations around current “elite” points, and keeps the best
# individuals each iteration. It uses one-dimensional step-size control and
# bounds-aware clipping to handle constraints.
#
# Search state: The algorithm tracks current best solution (best_x, best_y),
# a population of candidate vectors, and a global mutation scale (sigma).
# It also tracks evaluations consumed so it never exceeds the given budget.
#
# Candidate generation: Each iteration creates offspring by selecting parents
# from the current population (favoring better candidates) and adding
# sigma-scaled Gaussian noise. A fraction of offspring are “jittered”
# around the current best to encourage exploitation.
#
# Selection and replacement: The objective values are evaluated for offspring.
# A (μ + λ) style replacement is used: offspring are merged with the best
# portion of the population, and the next population is formed by taking the
# lowest objective values (minimization).
#
# Adaptation: Sigma is adapted based on a success measure: after each
# iteration, if offspring improve the current best, sigma is mildly decreased
# (or kept), otherwise sigma is mildly increased to regain exploration.
#
# Exploration mechanisms: Larger sigma and parent sampling with probabilities
# biased toward better individuals encourage global exploration early, while
# periodic jitter around the best provides local exploration.
#
# Exploitation mechanisms: A fraction of offspring are generated around the
# current best solution. When improvements occur, sigma reduction tightens
# the search around promising regions.
#
# Boundary handling: New candidates are clipped to the provided bounds to
# ensure feasibility. Bounds are read from func.lower/func.upper or
# func.bounds.lb/func.bounds.ub.
#
# Budget strategy: The algorithm computes a safe number of iterations and a
# population/offpspring size such that total objective evaluations never exceed
# budget. It stops early if the budget would be exceeded.
#
# Closest known influences: The design resembles an evolution strategy /
# CMA-like simplified approach (DE/ES intuition), with success-based step-size
# control and elite-guided sampling, adapted to a strict evaluation budget.
#
# Failure modes: If the budget is extremely small, the algorithm falls back
# to sampling a tiny initial population and returns the best seen. Clipping
# can cause many candidates to stick on boundaries for ill-scaled problems.
# If the objective is very noisy or discontinuous, step-size adaptation may
# oscillate.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")

    def __call__(self, func):
        lb, ub = self._get_bounds(func)
        dim = self.dim

        # Defensive: ensure finite bounds; if bounds are missing or degenerate,
        # handle gracefully with zeros range.
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds must match the problem dimension.")

        # Ensure lb <= ub; if not, swap elementwise.
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        lb, ub = lo, hi

        # Range used for scaling mutations and initial sampling.
        span = ub - lb
        # Avoid zero span causing sigma=0 everywhere.
        safe_span = np.where(span > 0, span, 1.0)

        # Choose population size based on budget and dimension (compact + robust).
        # We must never exceed evaluation budget.
        # We'll use a (μ + λ) style: init μ, then per iteration evaluate λ.
        # Total evals = μ + iters*λ, stop early if necessary.
        mu = int(max(4, min(12, self.budget // 4)))  # elite pool size
        mu = min(mu, self.budget)  # cannot exceed budget
        # offspring per iteration
        lam = int(max(4, min(16, self.budget // 8)))
        # Cap offspring so that at least one iteration can happen.
        lam = min(lam, max(1, self.budget - mu)) if self.budget > mu else 0

        # If budget is too small, just do random samples and return best.
        if self.budget <= mu or lam == 0:
            n = min(self.budget, max(1, self.budget))
            best_x, best_y, _ = self._sample_and_eval(func, lb, ub, n)
            return best_x, best_y

        # Determine maximum number of iterations we can afford.
        remaining = self.budget - mu
        iters = max(1, remaining // lam)
        # We'll run at most `iters`, but can stop early if budget becomes tight.

        # Initialize population uniformly within bounds.
        pop = self._uniform_init(lb, ub, mu)
        f_pop = np.empty(mu, dtype=float)
        evals = 0
        for i in range(mu):
            f_pop[i] = float(func(pop[i]))
            evals += 1

        # Track current best.
        best_idx = int(np.argmin(f_pop))
        best_x = pop[best_idx].copy()
        best_y = float(f_pop[best_idx])

        # Step size initialization: fraction of bounds span.
        # Use global sigma (not per-dim) for compactness.
        sigma = 0.3 * np.mean(safe_span)
        # Also allow sigma to be small if mean span is very small.
        sigma = max(sigma, 1e-12)

        # Probability bias towards better parents: soft rank-based.
        ranks = np.argsort(np.argsort(f_pop))  # 0 best
        # Convert ranks to weights: higher for better individuals.
        # weights ~ (mu - rank), ensuring positivity.
        weights = (mu - ranks).astype(float)
        weights_sum = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0
        weights /= weights_sum

        for _ in range(iters):
            if evals >= self.budget:
                break

            # Ensure we don't exceed budget:
            # we may not have enough evals left for full lam.
            cur_lam = min(lam, self.budget - evals)
            if cur_lam <= 0:
                break

            # Generate offspring.
            # - Majority around elite parents with Gaussian noise
            # - Small fraction around best_x for exploitation.
            offspring = np.empty((cur_lam, dim), dtype=float)

            # Exploitation fraction:
            frac_best = 0.25
            n_best = int(round(frac_best * cur_lam))
            n_best = min(n_best, cur_lam)

            # Offspring around best.
            if n_best > 0:
                noise = np.random.randn(n_best, dim)
                # Scale noise by sigma and typical span to be scale-aware.
                # Use normalized span to reduce issues across dims.
                span_scale = safe_span / np.mean(safe_span)
                cand = best_x[None, :] + (sigma * noise) * span_scale[None, :]
                offspring[:n_best] = self._clip(cand, lb, ub)

            # Remaining offspring around parents.
            n_rest = cur_lam - n_best
            if n_rest > 0:
                # Sample parent indices with replacement using biased weights.
                parent_idx = np.random.choice(mu, size=n_rest, replace=True, p=weights)
                parents = pop[parent_idx]
                noise = np.random.randn(n_rest, dim)
                span_scale = safe_span / np.mean(safe_span)
                cand = parents + (sigma * noise) * span_scale[None, :]
                offspring[n_best:] = self._clip(cand, lb, ub)

            # Evaluate offspring.
            f_off = np.empty(cur_lam, dtype=float)
            for i in range(cur_lam):
                f_off[i] = float(func(offspring[i]))
                evals += 1

            # Update best.
            off_best_i = int(np.argmin(f_off))
            off_best_y = float(f_off[off_best_i])
            if off_best_y < best_y:
                best_y = off_best_y
                best_x = offspring[off_best_i].copy()

            # Adapt sigma: success-based.
            # If we made improvement, shrink; otherwise expand slightly.
            # The shrink/expand factors are mild to remain stable.
            if off_best_y < best_y + 0.0:  # This line is always false after assignment, so use previous best.
                pass
            # Instead, compute improvement relative to previous best stored before evaluation:
            # We'll recompute using f_pop/best_y? Simpler:
            # We'll interpret improvement by comparing min offspring to current best_y before update.
            # Because best_y has been updated, we need an approximate measure:
            # Use whether offspring min is strictly lower than the previous stored best
            # by tracking it explicitly.
            # To keep it correct, we capture previous best at start of iteration.
            # (We didn't; so we use a more robust criterion: compare with f_pop best.)
            # We'll instead compare with min(f_pop) which approximates previous best before offspring.
            # If offspring improves below prior elite best, treat as success.
            prior_elite_best = float(np.min(f_pop))
            if float(np.min(f_off)) < prior_elite_best:
                sigma *= 0.85
            else:
                sigma *= 1.05
            sigma = float(np.clip(sigma, 1e-12, 10.0 * np.mean(safe_span)))

            # Selection + replacement:
            # Merge a fraction of current population elites with offspring.
            elite_keep = max(2, mu // 2)
            elite_idx = np.argsort(f_pop)[:elite_keep]
            keep_pop = pop[elite_idx]
            keep_f = f_pop[elite_idx]

            merged_pop = np.vstack([keep_pop, offspring])
            merged_f = np.concatenate([keep_f, f_off])

            # Keep best mu.
            best_mu_idx = np.argsort(merged_f)[:mu]
            pop = merged_pop[best_mu_idx]
            f_pop = merged_f[best_mu_idx]

            # Update parent sampling weights based on new population.
            ranks = np.argsort(np.argsort(f_pop))
            weights = (mu - ranks).astype(float)
            weights_sum = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0
            weights /= weights_sum

        return best_x, best_y

    def _get_bounds(self, func):
        # Bounds may be exposed in two styles.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return np.asarray(func.lower, dtype=float), np.asarray(func.upper, dtype=float)
        if hasattr(func, "bounds"):
            b = func.bounds
            # Try lb/ub first; fall back to lower/upper.
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return np.asarray(b.lb, dtype=float), np.asarray(b.ub, dtype=float)
            if hasattr(b, "lower") and hasattr(b, "upper"):
                return np.asarray(b.lower, dtype=float), np.asarray(b.upper, dtype=float)
        raise AttributeError("Cannot read bounds from func.lower/upper or func.bounds.lb/ub")

    def _uniform_init(self, lb, ub, n):
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        # Handle potential degenerate spans.
        r = np.random.rand(n, lb.size)
        return lb[None, :] + r * (ub - lb)[None, :]

    def _clip(self, x, lb, ub):
        return np.minimum(np.maximum(x, lb[None, :] if x.ndim == 2 else lb), ub[None, :] if x.ndim == 2 else ub)

    def _sample_and_eval(self, func, lb, ub, n):
        pop = self._uniform_init(lb, ub, n)
        best_x = pop[0].copy()
        best_y = float(func(best_x))
        for i in range(1, n):
            y = float(func(pop[i]))
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()
        return best_x, best_y, n
