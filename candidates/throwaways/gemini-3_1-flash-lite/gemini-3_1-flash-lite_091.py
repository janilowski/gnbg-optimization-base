# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the Differential Evolution (DE/rand/1/bin) algorithm.
# Search state: A population of vectors of size (population_size, dimension).
# Candidate generation: Mutation using the difference vector of two random population members, followed by binomial crossover.
# Selection and replacement: Greedy selection where a mutant replaces the target vector only if it produces a lower objective value.
# Adaptation: None (fixed hyperparameters).
# Exploration mechanisms: Large-scale global search driven by the difference vector magnitude controlled by the mutation factor.
# Exploitation mechanisms: Crossover and greedy selection focusing on local improvements.
# Boundary handling: Clipping values to the defined lower/upper bounds.
# Budget strategy: The population size is scaled with dimension, and iterations continue until the evaluation budget is exhausted.
# Closest known influences: Storn and Price's Differential Evolution.
# Novelty or unusual aspects: Minimalist implementation suited for black-box benchmarks.
# Failure modes: Can get trapped in local optima for highly multimodal functions if the population diversity is lost too early.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Population size scaling logic: 10 + 2*dim, capped by budget/2
        self.pop_size = min(max(10, 2 * dim), budget // 2)
        self.mut = 0.8  # Differential weight
        self.crossp = 0.7  # Crossover probability

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        else:
            lb, ub = -5.12 * np.ones(self.dim), 5.12 * np.ones(self.dim)

        # Initialize population
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Iterative evolution
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break

                # Mutation: pick 3 distinct others
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Mutation and Crossover
                mutant = np.clip(a + self.mut * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                f = func(trial)
                evals += 1
                
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    if f < best_y:
                        best_y = f
                        best_x = trial.copy()

        return best_x, best_y
