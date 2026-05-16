# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An Ensemble Differential Evolution algorithm maintaining a pool of distinct mutation strategies and crossover parameter pairs.
# Search state: Retains active population positions, objective fitness values, strategy and parameter assignments per individual, and global optimum.
# Candidate generation: Generates trial vectors using individual strategy assignments (best/1, current-to-best, or rand/2) and assigned F/CR values.
# Selection and replacement: Replaces parent solutions with successful trial vectors; successful individuals retain their assigned strategy and parameters.
# Adaptation: Unsuccessful individuals re-sample their strategy and parameter pairs from the ensemble pool to dynamically find effective operators.
# Exploration mechanisms: Strategy rand/2 and parameter pairs with low CR maintain wide exploratory diversity across unvisited basins.
# Exploitation mechanisms: Strategies best/1 and current-to-best aggressively pull search vectors towards the elite global best.
# Boundary handling: All trial positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population trial vectors sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Ensemble of Mutation Strategies and Parameters in DE / EPSDE (Mallipeddi et al.).
# Novelty or unusual aspects: Directly embeds discrete strategy switching logic and parameter pairing inside the individual evaluation loop.
# Failure modes: Can waste evaluations on ineffective operators if the ensemble pool contains poorly suited strategies for the landscape.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 8, max(25, 2 * self.dim)))
        if self.pop_size > 60:
            self.pop_size = 60

    def __call__(self, func):
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        self.eval_count = 0
        domain_range = ub - lb

        best_x = None
        best_y = float("inf")

        pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        fitness = np.full(self.pop_size, float("inf"))

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop[i]))
            self.eval_count += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Strategy pool: 0 -> best/1, 1 -> current-to-best, 2 -> rand/2
        n_strategies = 3
        param_pool = [(0.5, 0.5), (0.8, 0.2), (0.9, 0.9), (0.4, 0.8), (0.6, 0.6)]
        n_params = len(param_pool)

        # Initial assignments
        strat_assign = np.random.randint(0, n_strategies, size=self.pop_size)
        param_assign = np.random.randint(0, n_params, size=self.pop_size)

        while self.eval_count < self.budget:
            best_idx = np.argmin(fitness)
            pop_best = pop[best_idx]

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                strat = strat_assign[i]
                f, cr = param_pool[param_assign[i]]

                if strat == 0:
                    # best/1
                    r1, r2 = np.random.choice(self.pop_size, size=2, replace=False)
                    v = pop_best + f * (pop[r1] - pop[r2])
                elif strat == 1:
                    # current-to-best
                    r1, r2 = np.random.choice(self.pop_size, size=2, replace=False)
                    v = pop[i] + f * (pop_best - pop[i]) + f * (pop[r1] - pop[r2])
                else:
                    # rand/2
                    r1, r2, r3, r4, r5 = np.random.choice(self.pop_size, size=5, replace=False)
                    v = pop[r1] + f * (pop[r2] - pop[r3]) + f * (pop[r4] - pop[r5])

                mask = np.random.rand(self.dim) <= cr
                j_rand = np.random.randint(self.dim)
                mask[j_rand] = True

                trial = np.where(mask, v, pop[i])
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y < fitness[i]:
                    pop[i] = trial
                    fitness[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()
                else:
                    # Re-assign strategy and parameters
                    strat_assign[i] = np.random.randint(0, n_strategies)
                    param_assign[i] = np.random.randint(0, n_params)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
