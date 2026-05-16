# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust Differential Evolution (DE) algorithm utilizing the DE/rand/1/bin mutation strategy with dithered scale factors.
# Search state: Maintains a population of candidate vectors and their corresponding fitness values, as well as the global best solution found.
# Candidate generation: Generates trial vectors by taking three random distinct population members, computing a scaled difference vector, and applying binomial crossover with the target vector.
# Selection and replacement: Evaluates the trial vector and replaces the parent if the trial vector's objective value is less than or equal to the parent's.
# Adaptation: Employs a dithered mutation scale factor F in each generation to enhance diversity and prevent premature convergence.
# Exploration mechanisms: Population diversity across the bounded domain and stochastic mutation steps based on vector differences provide robust global exploration.
# Exploitation mechanisms: Greedy selection guarantees monotonic improvement for each population slot, and binomial crossover preserves successful coordinate values.
# Boundary handling: All trial vectors are explicitly clipped to the search space lower and upper bounds before evaluation.
# Budget strategy: Allocates evaluations strictly one-by-one during population initialization and subsequent generational loops until the budget is exhausted.
# Closest known influences: Standard Differential Evolution (Storn & Price).
# Novelty or unusual aspects: Dynamically bounds population size based on search space dimensionality and evaluation budget to ensure sufficient generational turnover.
# Failure modes: Can struggle on extremely noisy landscapes or if the budget is too small relative to the required population size in high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 5, max(10, 3 * self.dim)))
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
        best_x = None
        best_y = float("inf")

        # Initialize population uniformly within bounds
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

        cr = 0.85  # Crossover probability

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Choose 3 distinct random members not equal to i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Dithered F factor between 0.5 and 0.9
                f = np.random.uniform(0.5, 0.9)
                mutant = pop[r1] + f * (pop[r2] - pop[r3])

                # Binomial crossover
                cross_mask = np.random.rand(self.dim) < cr
                # Ensure at least one dimension is crossed
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(self.dim)] = True

                trial = np.where(cross_mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y <= fitness[i]:
                    fitness[i] = y
                    pop[i] = trial
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
