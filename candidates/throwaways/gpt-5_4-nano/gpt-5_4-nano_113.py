import numpy as np


# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy using an
# adaptive population of candidate points and a (μ,λ)-style evolutionary
# loop with step-size adaptation. It maintains and refines a "best" solution
# while sampling new points around it, balancing exploration and exploitation.
# Search state: Keeps an array of candidate solutions, their fitness values,
# current best solution/value, an adaptive global step size (sigma), and an
# evaluation counter to strictly respect the budget.
# Candidate generation: Initializes a population by uniform sampling within
# the provided bounds. In each iteration, generates offspring by perturbing
# current elite points with Gaussian noise scaled by sigma, using different
# mutation directions per offspring.
# Selection and replacement: Evaluates offspring, then keeps the best μ points
# (elitism) from the union of parents and offspring, updating the global best.
# Adaptation: Updates sigma based on a simple success-rate heuristic derived
# from how many offspring improved upon the current best; sigma decreases
# when progress is scarce and increases slightly when many improvements occur.
# Exploration mechanisms: Periodic diversity refresh (random reinitialization of
# a fraction of the population) and occasional larger mutations when the search
# stagnates; also uses multiple elites as perturbation centers.
# Exploitation mechanisms: Gaussian sampling around elites with adaptive sigma
# encourages local refinement; elites are preserved each iteration.
# Boundary handling: After mutation, candidates are clipped to the valid bounds.
# Budget strategy: Computes a safe number of evaluations per phase and then
# runs iterations that never exceed the provided evaluation budget. All calls
# to the objective are counted; no extra evaluations occur.
# Closest known influences: A mix of (μ,λ)-evolutionary strategies, success-based
# step-size adaptation, and elitist selection; similar in spirit to CMA-ES-like
# heuristics but intentionally kept simple and lightweight.
# Novelty or unusual aspects: Uses bound-aware scaling for sigma initialization
# and a lightweight stagnation-triggered population diversity refresh rather
# than full covariance modeling.
# Failure modes: For very noisy or deceptive objectives, the success-based
# sigma update may oscillate; for extremely tight or ill-conditioned bounds,
# clipping can reduce effective exploration. The method still remains within
# the evaluation budget.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            raise ValueError("budget must be positive")

        # Read bounds from func
        lower = None
        upper = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lower = np.asarray(b.lb, dtype=float)
                upper = np.asarray(b.ub, dtype=float)
        if lower is None or upper is None:
            raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/ub")

        if lower.shape != (dim,) or upper.shape != (dim,):
            lower = lower.reshape((dim,))
            upper = upper.reshape((dim,))

        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("Bounds must be finite arrays")

        lo = lower
        hi = upper
        span = hi - lo
        # Guard against degenerate dimensions
        span_safe = np.where(span > 0, span, 1.0)

        evals = 0

        def eval_obj(x):
            nonlocal evals
            if evals >= budget:
                # Should never happen; defensive.
                return np.inf
            y = func(x)
            evals += 1
            return float(y)

        # Helper: clip to bounds
        def clip_to_bounds(x):
            return np.minimum(hi, np.maximum(lo, x))

        # Population parameters (μ, λ) with sensible scaling across dimensions.
        # Ensure at least a small population, but never exceed budget.
        mu = max(2, min(10 + dim // 2, budget // 2 if budget >= 2 else 1))
        lam = max(2, min(20 + dim, max(2, budget - mu) if budget > mu else 2))

        # If budget is extremely small, fall back to pure random search.
        if budget <= 3 or mu + lam > budget:
            k = min(budget, max(1, budget))
            best_x = None
            best_y = np.inf
            # Initialize best with one random point.
            for _ in range(k):
                x = lo + np.random.random(dim) * span_safe
                x = clip_to_bounds(x)
                y = eval_obj(x)
                if y < best_y or best_x is None:
                    best_y, best_x = y, x.copy()
            return best_x, best_y

        # Initial step size: fraction of typical span (bound-aware).
        sigma = 0.25 * np.median(span_safe)
        sigma = float(sigma) if sigma > 0 else 0.25

        # Initialize population uniformly in bounds.
        pop = lo + np.random.random((mu, dim)) * span_safe
        pop = clip_to_bounds(pop)

        fit = np.empty(mu, dtype=float)
        for i in range(mu):
            fit[i] = eval_obj(pop[i])

        best_idx = int(np.argmin(fit))
        best_x = pop[best_idx].copy()
        best_y = float(fit[best_idx])

        # Iteration budget tracking:
        # Each loop evaluates λ offspring. We'll run as many full loops as possible.
        # Note: mu evaluations already consumed.
        remaining = budget - evals
        if remaining <= 0:
            return best_x, best_y

        max_iters = max(0, remaining // lam)
        # Use at most one partial iteration if budget allows some offspring.
        partial_offspring = remaining - max_iters * lam

        def choose_elites(k):
            # Return indices of k best individuals.
            kk = min(k, mu)
            return np.argpartition(fit, kk - 1)[:kk]

        stagnation = 0
        prev_best = best_y

        # Diversity refresh fraction when stuck.
        refresh_frac = 0.2 if dim >= 2 else 0.3
        # Number of elites used as mutation centers.
        elite_k = max(2, min(mu, 3 + dim // 5))

        total_iters = max_iters + (1 if partial_offspring > 0 else 0)

        for it in range(total_iters):
            # Determine how many offspring we can evaluate this iteration.
            cur_lam = lam if it < max_iters else partial_offspring
            if cur_lam <= 0:
                break

            elite_ids = choose_elites(elite_k)
            elites = pop[elite_ids]

            # Generate offspring:
            # - Pick elite center per offspring
            # - Add Gaussian noise with sigma scaled by per-dim span
            # - Small additional jitter on top encourages exploration
            # This is isotropic but span-scaled for robustness.
            span_scale = span_safe / (np.sqrt(np.mean(span_safe ** 2)) + 1e-12)
            # Build mutation scales with span-aware normalization.
            mutation_scale = sigma * span_scale

            offspring = np.empty((cur_lam, dim), dtype=float)
            # Randomly assign each offspring to an elite center
            centers = elites[np.random.randint(0, elites.shape[0], size=cur_lam)]
            noise = np.random.randn(cur_lam, dim) * mutation_scale
            offspring = centers + noise

            # Occasionally, enlarge mutations to escape stagnation.
            if stagnation >= 2 and (np.random.rand() < 0.5):
                offspring += np.random.randn(cur_lam, dim) * (0.75 * sigma * span_scale)

            offspring = clip_to_bounds(offspring)

            # Evaluate offspring strictly within budget.
            off_fit = np.empty(cur_lam, dtype=float)
            for j in range(cur_lam):
                off_fit[j] = eval_obj(offspring[j])

            # Combine and select best μ
            combined_pop = np.vstack((pop, offspring))
            combined_fit = np.concatenate((fit, off_fit))
            # Select indices of μ best
            sel = np.argpartition(combined_fit, mu - 1)[:mu]
            pop = combined_pop[sel]
            fit = combined_fit[sel]

            cur_best_idx = int(np.argmin(fit))
            cur_best_y = float(fit[cur_best_idx])
            cur_best_x = pop[cur_best_idx].copy()

            # Update global best
            if cur_best_y < best_y:
                best_y = cur_best_y
                best_x = cur_best_x
                stagnation = 0
            else:
                stagnation += 1

            # Success-rate heuristic for sigma adaptation:
            # Measure how many offspring improved upon previous best.
            improvements = int(np.sum(off_fit < prev_best))
            success_rate = improvements / max(1, cur_lam)

            # Adapt sigma: decrease if low success, increase if high success
            # Use gentle multiplicative updates to avoid instability.
            if success_rate > 0.2:
                sigma *= 1.07
            elif success_rate < 0.05:
                sigma *= 0.90
            else:
                sigma *= 1.00

            # Additional stagnation-based refresh & sigma bump
            if stagnation >= 4:
                # Refresh a fraction of the population uniformly within bounds.
                r = max(1, int(refresh_frac * mu))
                ids = np.random.choice(mu, size=r, replace=False)
                new_pts = lo + np.random.random((r, dim)) * span_safe
                new_pts = clip_to_bounds(new_pts)
                # Evaluate refreshed individuals only if budget allows;
                # otherwise, just keep them without evaluation (can't happen ideally
                # because we already respect remaining budget).
                for ii in range(r):
                    if evals >= budget:
                        break
                    idx = ids[ii]
                    pop[idx] = new_pts[ii]
                    fit[idx] = eval_obj(pop[idx])
                    if fit[idx] < best_y:
                        best_y = float(fit[idx])
                        best_x = pop[idx].copy()

                # Inflate sigma slightly after refresh to encourage re-exploration.
                sigma *= 1.15
                # Reset stagnation after refresh attempt.
                stagnation = 0

            prev_best = min(prev_best, cur_best_y)

            # Keep sigma within reasonable range based on bounds.
            # Upper bound: span median; lower bound: tiny fraction of span.
            span_med = float(np.median(span_safe))
            sigma = float(np.clip(sigma, 1e-12 * max(1.0, span_med), max(1e-12, span_med)))

        return best_x, best_y
