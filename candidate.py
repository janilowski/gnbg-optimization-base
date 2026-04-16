from __future__ import annotations

import numpy as np


class Algorithm:
    """
    Population-based optimization with budget-safe local refinement.
    Interface expected by the harness:
      - __init__(self, budget, dim)
      - __call__(self, func)
    """

    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lower = np.asarray(getattr(func, "lower", func.bounds.lb), dtype=float)
        upper = np.asarray(getattr(func, "upper", func.bounds.ub), dtype=float)
        lower = np.broadcast_to(lower, (self.dim,)).astype(float)
        upper = np.broadcast_to(upper, (self.dim,)).astype(float)

        # Conservative population size that still leaves room for iterations.
        pop_size = max(8, min(28, 4 * self.dim))
        if self.budget <= pop_size:
            pop_size = max(1, self.budget)

        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= self.budget:
                return None
            y = float(func(x))
            evals += 1
            return y

        # Initial population: uniform + one center sample when possible.
        pop = np.random.uniform(lower, upper, size=(pop_size, self.dim))
        if pop_size >= 2:
            pop[0] = 0.5 * (lower + upper)

        fit = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            y = evaluate(pop[i])
            if y is None:
                # Very small budget edge case.
                best_idx = int(np.argmin(fit[:i])) if i > 0 else 0
                return pop[best_idx].copy(), float(fit[best_idx]) if i > 0 else float("inf")
            fit[i] = y

        best_idx = int(np.argmin(fit))
        best_x = pop[best_idx].copy()
        best_y = float(fit[best_idx])

        # Differential-evolution style global search + occasional local steps.
        while evals < self.budget:
            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # Distinct indices for mutation.
                idxs = np.arange(pop_size)
                idxs = idxs[idxs != i]
                if idxs.size < 3:
                    break
                r1, r2, r3 = np.random.choice(idxs, size=3, replace=False)

                # Mildly jittered parameters for robustness across landscapes.
                F = float(np.clip(0.6 + 0.3 * np.random.randn(), 0.3, 0.9))
                CR = float(np.clip(0.85 + 0.1 * np.random.randn(), 0.5, 0.98))

                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lower, upper)

                # Binomial crossover with forced crossover dimension.
                trial = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                cross_mask = np.random.rand(self.dim) < CR
                cross_mask[j_rand] = True
                trial[cross_mask] = mutant[cross_mask]

                # Occasional local perturbation around global best.
                if np.random.rand() < 0.15:
                    span = (upper - lower)
                    sigma = (0.12 + 0.25 * (1.0 - evals / max(1, self.budget))) * span
                    local = best_x + np.random.randn(self.dim) * sigma
                    local = np.clip(local, lower, upper)
                    trial = 0.5 * trial + 0.5 * local

                y_trial = evaluate(trial)
                if y_trial is None:
                    break

                if y_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = y_trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()

        return best_x, best_y
