# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Differential Evolution algorithm weighting base vector selection probabilities according to objective fitness rankings.
# Search state: Retains active population positions, objective fitness values, and global optimum across generational iterations.
# Candidate generation: Generates trial vectors via rank-selected base vectors combined with uniformly sampled difference vectors.
# Selection and replacement: Standard one-to-one parent replacement; re-sorts population each generation to update selection probabilities.
# Adaptation: Automatically biases search directions towards superior solutions without relying on a single absolute global best anchor.
# Exploration mechanisms: Uniform random selection of difference vector pairs (r2, r3) maintains mutational step magnitude diversity.
# Exploitation mechanisms: Linear rank-based roulette selection ensures highly fit individuals frequently serve as mutation base anchors.
# Boundary handling: All trial vector positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population trial vectors sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Rank-Based Differential Evolution / Rank-DE (Gong et al.).
# Novelty or unusual aspects: Directly embeds pre-computed linear ranking selection probabilities into base vector indexing.
# Failure modes: Can experience premature variance collapse if ranking weights are excessively steep in early generations.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 8, max(20, 2 * self.dim)))
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

        # Precompute linear rank probabilities
        ranks = np.arange(self.pop_size)
        weights = 2.0 * (self.pop_size - ranks) / (self.pop_size * (self.pop_size + 1.0))
        probs = weights / np.sum(weights)

        f, cr = 0.8, 0.7

        while self.eval_count < self.budget:
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]

            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Base vector r1 selected via rank roulette wheel
                r1 = np.random.choice(self.pop_size, p=probs)
                
                r2 = np.random.randint(self.pop_size)
                while r2 == r1:
                    r2 = np.random.randint(self.pop_size)

                r3 = np.random.randint(self.pop_size)
                while r3 == r1 or r3 == r2:
                    r3 = np.random.randint(self.pop_size)

                v = pop[r1] + f * (pop[r2] - pop[r3])
                mask = np.random.rand(self.dim) <= cr
                mask[np.random.randint(self.dim)] = True
                trial = np.clip(np.where(mask, v, pop[i]), lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y <= fitness[i]:
                    next_pop[i] = trial
                    next_fit[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()
                else:
                    next_pop[i] = pop[i]
                    next_fit[i] = fitness[i]

            pop = next_pop
            fitness = next_fit

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
