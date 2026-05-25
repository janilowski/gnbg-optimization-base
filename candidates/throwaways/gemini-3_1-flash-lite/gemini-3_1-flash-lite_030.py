# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the Differential Evolution (DE/rand/1/bin) algorithm.
# Search state: Maintains a population of candidate solutions of size 10 + 2*dim.
# Candidate generation: Uses mutation (vector difference) and binomial crossover to create offspring.
# Selection and replacement: Greedy selection: offspring replace parents only if they yield a lower objective value.
# Adaptation: None (fixed mutation/crossover parameters).
# Exploration mechanisms: Large-scale movement via differential vectors (population spread).
# Exploitation mechanisms: Greedy selection and recombination around existing good points.
# Boundary handling: Clamping to provided lb/ub bounds.
# Budget strategy: Stops exactly when the population evaluation count hits the provided budget.
# Closest known influences: Storn and Price's Differential Evolution.
# Novelty or unusual aspects: Minimalist implementation focused on zero external dependencies.
# Failure modes: Can stall in local minima for highly multimodal or ill-conditioned functions; sensitive to scaling.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(10, 2 * dim)
        self.f = 0.8  # Differential weight
        self.cr = 0.9 # Crossover probability

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialize population
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        y = np.array([func(x) for x in pop])
        
        evals = self.pop_size
        best_idx = np.argmin(y)
        best_x = pop[best_idx].copy()
        best_y = y[best_idx]

        # Optimization loop
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Mutation: pick 3 distinct random agents excluding current
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Mutation and Crossover
                mutant = a + self.f * (b - c)
                mutant = np.clip(mutant, lb, ub)
                
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                trial_y = func(trial)
                evals += 1
                
                if trial_y <= y[i]:
                    pop[i] = trial
                    y[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()
        
        return best_x, best_y
