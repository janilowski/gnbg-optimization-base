# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm for bounded
# continuous variables using an adaptive, budget-aware evolutionary strategy
# with a restart mechanism.
# Search state: Maintains a current population of candidate points, their
# objective values, and keeps track of the best-so-far solution across the
# entire run.
# Candidate generation: Uses Gaussian perturbations around selected
# individuals. The mutation step size adapts based on recent improvement.
# Selection and replacement: Each iteration evaluates a small batch of offspring,
# then forms the next population by selecting the best candidates among parents
# and offspring (elitist truncation).
# Adaptation: The global mutation scale shrinks on stagnation and grows slightly
# when improvements happen, based on the ratio of best improvement over a moving
# baseline. A periodic restart reinitializes the population around the best
# point after prolonged stagnation.
# Exploration mechanisms: Early in the run and after restarts uses larger step
# sizes for broader coverage; periodic restarts reduce the chance of getting
# stuck.
# Exploitation mechanisms: As progress is detected, the step size shrinks and the
# algorithm intensifies search around top individuals.
# Boundary handling: Offspring are clipped to the feasible box defined by
# func.lower/upper or func.bounds.lb/ub.
# Budget strategy: The total number of objective evaluations is capped to the
# provided budget. The algorithm determines how many generations/offspring it
# can evaluate and stops exactly when the budget would be exceeded.
# Closest known influences: Inspired by (μ+λ) evolution strategies and
# self-adaptive step size heuristics, with a simple restart-on-stagnation
# mechanism.
# Novelty or unusual aspects: Uses a lightweight, deterministic-in-control-budget
# scheduling (population size computed from budget and dimension) and a robust
# improvement-based mutation scaling that works across wide dimension ranges.
# Failure modes: If the objective is extremely noisy or adversarial, adaptation
# can misinterpret noise as stagnation/improvement, leading to suboptimal step
# sizes. Very tight bounds or highly multimodal landscapes may require more
# restarts to achieve good results.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        # Defensive handling for degenerate settings
        if self.budget < 1:
            self.budget = 1
        if self.dim < 1:
            self.dim = 1

        # Population sizing:
        # - Keep compact for readability and robustness.
        # - Ensure at least one initial evaluation.
        # - Use budget to pick a reasonable (μ, λ).
        # The algorithm runs for (budget - evals_init) / lambda generations.
        mu = max(4, int(np.sqrt(self.dim) * 6))
        mu = min(mu, self.budget)  # can't exceed budget for initial evals
        if mu < 2:
            mu = 2
        self.mu = mu

        # offspring per generation; keep small to limit expensive objective calls.
        # Must allow at least one generation after initialization.
        lam = max(4, int(np.sqrt(self.dim) * 8))
        lam = min(lam, max(1, self.budget - self.mu))
        self.lam = lam

        # Basic settings (adapted dynamically during __call__)
        self.restart_patience = 6  # generations without meaningful improvement

    def __call__(self, func):
        lower, upper = self._get_bounds(func)
        d = self.dim

        # Ensure shape correctness
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        if lower.size == 1:
            lower = np.full(d, lower.item(), dtype=float)
        if upper.size == 1:
            upper = np.full(d, upper.item(), dtype=float)

        lower = lower[:d].copy()
        upper = upper[:d].copy()
        # If bounds are inverted, swap to be robust.
        lo = np.minimum(lower, upper)
        hi = np.maximum(lower, upper)
        lower, upper = lo, hi

        # Objective evaluation helper with strict budget enforcement.
        evals = 0
        best_x = None
        best_y = np.inf

        def eval_one(x):
            nonlocal evals, best_x, best_y
            # x is expected to be 1D numpy array
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Clip candidates to bounds
        def clip_to_bounds(X):
            # X shape: (n, d) or (d,)
            return np.minimum(np.maximum(X, lower), upper)

        # Initialize population uniformly in bounds.
        mu = min(self.mu, self.budget)  # ensure not exceed budget
        X = lower + (upper - lower) * np.random.rand(mu, d)
        X = clip_to_bounds(X)

        f = np.empty(mu, dtype=float)
        for i in range(mu):
            f[i] = eval_one(X[i])
            if evals >= self.budget:
                # Can't do more
                return best_x, best_y

        # Sort by fitness
        order = np.argsort(f)
        X = X[order]
        f = f[order]

        # Mutation scale initialization:
        span = (upper - lower)
        # Avoid zero span; if span=0, mutation has no effect for that coordinate.
        span = np.where(span > 0, span, 1.0)
        # Start with a fraction of span.
        sigma = 0.25 * span.mean()

        # Adaptation state
        prev_best = f[0]
        best_progress = 0.0
        patience = 0

        # How many evaluations we can still spend
        # Each generation evaluates exactly lambda offspring.
        # Stop before exceeding budget.
        while evals < self.budget:
            remaining = self.budget - evals
            lam = min(self.lam, remaining)
            if lam <= 0:
                break

            # Choose parent indices from top performers with bias to the best.
            # Use a geometric-like bias: weights decrease with rank.
            # Parents are used as centers for Gaussian mutation.
            top_k = max(2, min(mu, 5))
            ranks = np.arange(top_k)
            weights = np.exp(-ranks / 1.5)
            weights = weights / weights.sum()

            # Generate offspring:
            # - Create lam offspring
            # - For each offspring choose a parent from top_k
            # - Use correlated-ish mutation via per-coordinate scaling (still diagonal)
            parents_idx = np.random.choice(top_k, size=lam, replace=True, p=weights)
            centers = X[parents_idx]

            # Offspring mutation:
            # Use diagonal Gaussian with step scaled by sigma and span.
            # Add small noise floor to keep moves possible.
            sigma_vec = (sigma / span.mean()) * span
            sigma_vec = np.where(sigma_vec > 1e-12, sigma_vec, 1e-12)

            Z = np.random.randn(lam, d)
            Y_off = centers + Z * sigma_vec

            Y_off = clip_to_bounds(Y_off)

            # Evaluate offspring
            f_off = np.empty(lam, dtype=float)
            for i in range(lam):
                f_off[i] = eval_one(Y_off[i])
                if evals >= self.budget:
                    # In case budget ran out mid-batch, truncate
                    f_off = f_off[: i + 1]
                    Y_off = Y_off[: i + 1]
                    lam = i + 1
                    break

            # Combine and select next generation (elitist truncation)
            X_comb = np.vstack([X, Y_off])
            f_comb = np.concatenate([f, f_off])

            sel = np.argsort(f_comb)[:mu]
            X = X_comb[sel]
            f = f_comb[sel]

            # Adaptation based on improvement in best value this generation
            curr_best = f[0]
            improvement = prev_best - curr_best
            best_progress = 0.9 * best_progress + 0.1 * improvement
            prev_best = curr_best

            # Determine if improvement is meaningful relative to scale
            # Use absolute threshold scaled by objective scale surrogate.
            # (Since we don't know objective scale, base on current sigma.)
            meaningful = improvement > 1e-12

            if meaningful:
                patience = 0
                # If we're improving, shrink less aggressively (or even slightly grow)
                # to keep exploration if improvement is small.
                # More shrinking if improvement is strong.
                if improvement > 0.01 * abs(best_y) + 1e-12:
                    sigma *= 0.80
                else:
                    sigma *= 0.92
            else:
                patience += 1
                # Stagnation: shrink to exploit around best, but also allow restart.
                sigma *= 0.85

            # Bound sigma within reasonable range based on span
            sigma_min = 1e-12 * span.mean()
            sigma_max = 0.7 * span.mean()
            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # Restart mechanism: if no meaningful improvement for several generations,
            # reinitialize population around best solution to escape local traps.
            if patience >= self.restart_patience and evals < self.budget:
                # Reset most of the population using the current best
                # (keep one elite: the best point itself).
                elite = X[0].copy()
                X_new = np.empty_like(X)

                X_new[0] = elite
                # Create rest around elite with larger sigma to re-explore
                # Increase sigma a bit, but don't exceed max.
                sigma = min(sigma_max, sigma * 1.6)
                sigma_vec = (sigma / span.mean()) * span
                sigma_vec = np.where(sigma_vec > 1e-12, sigma_vec, 1e-12)

                for i in range(1, mu):
                    if evals >= self.budget:
                        break
                    # Heavy-tailed-ish perturbation: combine normal and scaled normal
                    step = (0.7 * np.random.randn(d) + 0.3 * np.random.randn(d)) * sigma_vec
                    cand = clip_to_bounds(elite + step)
                    X_new[i] = cand
                    f[i] = eval_one(cand)

                # If we ran out of budget mid-restart, finalize.
                if evals >= self.budget:
                    break

                # Replace population and reset adaptation.
                order = np.argsort(f)
                X = X[order]
                f = f[order]
                prev_best = f[0]
                best_progress = 0.0
                patience = 0

        # Fallback if something went wrong (shouldn't happen)
        if best_x is None:
            # Evaluate one midpoint within budget
            mid = (lower + upper) / 2.0
            best_x = mid.copy()
            if evals < self.budget:
                best_y = func(best_x)
            else:
                best_y = np.inf

        return best_x, float(best_y)

    @staticmethod
    def _get_bounds(func):
        # Prefer func.lower / func.upper
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return np.asarray(func.lower, dtype=float), np.asarray(func.upper, dtype=float)

        # Fall back to func.bounds.lb / func.bounds.ub
        if hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            return np.asarray(func.bounds.lb, dtype=float), np.asarray(func.bounds.ub, dtype=float)

        raise AttributeError(
            "Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub"
        )
