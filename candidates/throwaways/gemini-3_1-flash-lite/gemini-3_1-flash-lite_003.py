# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the Differential Evolution (DE/rand/1/bin) algorithm.
# Search state: Maintains a population of candidate solutions of size 10 + 2*dim.
# Candidate generation: Generates trial vectors using mutation (random selection) and binomial crossover.
# Selection and replacement: Greedy selection; a trial vector replaces its parent if it yields a lower objective value.
# Adaptation: None; uses fixed standard DE hyperparameters (F=0.5, CR=0.9).
# Exploration mechanisms: Differential mutation allows global search by scaling differences between random population members.
# Exploitation mechanisms: Greedy replacement directs the population toward local minima.
# Boundary handling: Clamping; candidates outside bounds are projected back to the boundary.
# Budget strategy: Iterative generation until the function evaluation budget is exhausted.
# Closest known influences: Storn and Price's Differential Evolution.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness in black-box settings.
# Failure modes: May converge prematurely on highly multimodal landscapes or struggle with very high-dimensional ill-conditioned problems.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
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

        # Differential Evolution Parameters
        F = 0.5
        CR = 0.9

        # Optimization loop
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Mutation: select 3 distinct individuals excluding current i
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                
                # Mutation and Crossover
                mutant = a + F * (b - c)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary clamping
                trial = np.clip(trial, lb, ub)
                
                # Greedy Selection
                trial_y = func(trial)
                evals += 1
                
                if trial_y <= y[i]:
                    pop[i] = trial
                    y[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()
        
        return best_x, best_y
