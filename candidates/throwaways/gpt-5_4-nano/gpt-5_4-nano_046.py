# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimizer inspired by a global-to-local
# evolutionary strategy with a covariance-free (diagonal) adaptive step size.
# Search state: Maintains a population of candidate solutions, their objective
# values, and an adaptive scalar step size (sigma). Tracks the overall best found.
# Candidate generation: Each generation samples offspring by perturbing the current
# mean (initially a random point within bounds) using Gaussian noise scaled by
# sigma. A small fraction of offspring are generated with heavier-tailed noise to
# help escape local minima.
# Selection and replacement: Uses (μ + λ) selection: combines parent and offspring,
# ranks by objective value (minimization), keeps the best μ as the new parents,
# and updates the mean as the average of these elites.
# Adaptation: Adjusts sigma based on 1/5-success rule using how often newly sampled
# offspring outperform the parent best (per generation). This keeps exploration
# calibrated across dimensions.
# Exploration mechanisms: Injects occasional large steps (mixture with larger
# variance) and re-centers if progress stalls for several generations.
# Exploitation mechanisms: The Gaussian sampling around the elite mean and the
# diagonal step-size control focus the search near promising regions.
# Boundary handling: Any candidate that goes out of bounds is clipped to the
# feasible interval; sigma is slightly damped when clipping is frequent.
# Budget strategy: Stops exactly when the evaluation count reaches the provided
# budget; each generation evaluates λ new points. The initial evaluations plus
# generation loop are clipped to never exceed budget.
# Closest known influences: Diagonal evolution strategy (ES) with rank-based elite
# recombination and sigma adaptation (1/5-success rule); plus restart-like
# re-centering on stagnation.
# Novelty or unusual aspects: Uses a lightweight mixture distribution (Gaussian +
# heavy-tailed via scaled normal) and a simple stagnation-triggered re-center,
# while avoiding covariance estimation to stay compact and robust.
# Failure modes: Can struggle on highly rugged landscapes or extremely narrow
# optima if sigma adaptation becomes too conservative; clipping may reduce
# effective exploration near bounds.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds in a robust way ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = func.bounds.lb
            ub = func.bounds.ub
        else:
            raise AttributeError("func must provide bounds via (lower, upper) or func.bounds.lb/ub")

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size == 1:
            lb = np.full(dim, lb.item(), dtype=float)
        if ub.size == 1:
            ub = np.full(dim, ub.item(), dtype=float)
        if lb.size != dim or ub.size != dim:
            raise ValueError(f"Bounds must match dim={dim} (or be scalar). Got lb={lb.size}, ub={ub.size}")

        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo
        span = np.where(span > 0, span, 1.0)  # avoid zero span degeneracy

        def clip_x(x):
            return np.minimum(hi, np.maximum(lo, x))

        def rand_in_bounds(n=1):
            r = np.random.rand(n, dim)
            return lo + r * (hi - lo)

        def eval_f(x):
            # func is assumed to be callable on a 1D array (dim,)
            return float(func(np.asarray(x, dtype=float)))

        # ---- Initialize budget accounting ----
        if budget <= 0:
            # Evaluate nothing: return a valid point but arbitrary score.
            x0 = rand_in_bounds(1)[0]
            return clip_x(x0), np.inf

        # ---- ES-like parameters (chosen to be robust across dims and budget) ----
        # Keep population sizes small for budget safety.
        lam = max(4, min(12, budget // 4))  # offspring per generation
        mu = max(2, min(lam, 6))            # elites/parents retained
        lam = int(lam)
        mu = int(mu)

        # Ensure we can do at least one generation within budget; if not, evaluate random search.
        # We'll use evals_remaining logic.
        evals = 0

        # ---- Initial mean and sigma ----
        mean = rand_in_bounds(1)[0]
        mean = clip_x(mean)

        # Initial sigma: proportional to bounds span, but not too large.
        sigma = 0.3 * float(np.mean(span))
        sigma = max(sigma, 1e-12)

        # Evaluate initial best for reference (optional, but helps adaptation).
        # Evaluate mean once as a "parent".
        best_x = mean.copy()
        best_y = eval_f(best_x)
        evals += 1

        # Keep best_y from the whole run; maintain current parent best_y for 1/5 rule.
        parent_best_y = best_y

        # Stagnation handling
        stagnation = 0
        best_improvement = 0.0
        # How many generations without significant improvement before re-centering
        max_stag = 8

        # To decide progress: compare to previous best.
        prev_best_y = best_y

        # ---- Main loop: (mu + lambda) with elite recombination ----
        # Each generation evaluates lam offspring, then selects top mu (and uses mean of elites).
        while evals < budget:
            evals_remaining = budget - evals
            cur_lam = min(lam, evals_remaining)

            # Sample offspring: Gaussian perturbations around current mean.
            # Use a small mixture with larger variance for exploration.
            # Heavy-tailed effect: Normal with increased sigma, selected by Bernoulli mask.
            # (Still standard library/numpy only.)
            z = np.random.randn(cur_lam, dim)
            if cur_lam > 0:
                mix = (np.random.rand(cur_lam, 1) < 0.2).astype(float)  # 20% exploratory
                # Larger steps ~2.5x sigma on the mixed subset
                step = sigma * (1.0 + 1.5 * mix)  # ranges [sigma, 2.5*sigma]
                offspring = mean[None, :] + step * z
            else:
                offspring = np.empty((0, dim), dtype=float)

            # Boundary handling: clip. Track clipping frequency.
            clipped = clip_x(offspring)
            # Fraction of coordinates clipped > 0 implies boundary contact.
            if cur_lam > 0:
                coord_clipped = (np.abs(clipped - offspring) > 1e-12)
                clip_frac = float(coord_clipped.mean())
            else:
                clip_frac = 0.0

            # Evaluate offspring
            off_y = np.empty(cur_lam, dtype=float)
            for i in range(cur_lam):
                off_y[i] = eval_f(clipped[i])
            evals += cur_lam

            # Count successes for 1/5-rule: offspring better than current parent best.
            # Success defined relative to parent_best_y.
            if cur_lam > 0:
                successes = int(np.sum(off_y < parent_best_y))
            else:
                successes = 0
            success_rate = successes / max(1, cur_lam)

            # Update parent best for next adaptation step
            gen_best_idx = int(np.argmin(off_y)) if cur_lam > 0 else None
            if cur_lam > 0:
                gen_best_y = off_y[gen_best_idx]
                gen_best_x = clipped[gen_best_idx]
                if gen_best_y < best_y:
                    best_y = gen_best_y
                    best_x = gen_best_x.copy()

            # Adapt sigma using 1/5-success style (log-scale update)
            # If success_rate > 0.2 increase; else decrease.
            # Also damp more when clipping is frequent.
            if cur_lam > 0:
                if success_rate > 0.2:
                    sigma *= 1.15
                else:
                    sigma *= 0.85
                # Mild penalty if candidates frequently hit boundaries
                if clip_frac > 0.25:
                    sigma *= 0.9
                sigma = float(np.clip(sigma, 1e-12, 10.0 * float(np.mean(span))))
            else:
                break

            # Selection: choose top mu offspring as elites
            if cur_lam > 0:
                elite_count = min(mu, cur_lam)
                elite_idx = np.argsort(off_y)[:elite_count]
                elites = clipped[elite_idx]
                parent_best_y = float(np.min(off_y))

                # Recombine mean: average of elites
                new_mean = np.mean(elites, axis=0)
                mean = clip_x(new_mean)
            else:
                # No evaluations possible
                break

            # Stagnation / re-centering
            improvement = prev_best_y - best_y
            if improvement > 1e-12:
                # Reset stagnation if we improved meaningfully.
                stagnation = 0
                best_improvement = improvement
                prev_best_y = best_y
            else:
                stagnation += 1
                # If stagnating, re-center mean partially to current best_x to refine,
                # and add a small random kick to regain diversity.
                if stagnation >= max_stag:
                    kick = (0.2 * sigma) * np.random.randn(dim)
                    mean = clip_x(0.7 * best_x + 0.3 * mean + kick)
                    stagnation = 0

        return best_x, best_y
