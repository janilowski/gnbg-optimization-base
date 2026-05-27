# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A minimalist implementation of the Differential Evolution algorithm (DE/rand/1/bin).
# Search state: Maintains a population of candidate vectors.
# Candidate generation: Creates trial vectors by adding the weighted difference of two random population members to a third.
# Selection and replacement: Uses greedy selection; a trial vector replaces the target if it yields a lower objective value.
# Adaptation: None; mutation and crossover factors are fixed.
# Exploration mechanisms: Differential mutation and stochastic crossover provide global exploration.
# Exploitation mechanisms: Greedy selection refines the population toward regions of lower cost.
# Boundary handling: Clamping to provided lower and upper bounds.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: Storn and Price's Differential Evolution.
# Novelty or unusual aspects: Compact, zero-dependency, and robust initialization.
# Failure modes: Can stall on extremely high-dimensional or highly non-separable non-convex landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(40, 10 + 2 * dim)
        self.f = 0.8  # Mutation factor
        self.cr = 0.9  # Crossover probability

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
        best_x, best_y = pop[best_idx].copy(), y[best_idx]

        # Optimization loop
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break

                # Mutation: select three distinct individuals
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Differential evolution mutation
                mutant = a + self.f * (b - c)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary clamping
                trial = np.clip(trial, lb, ub)
                
                # Selection
                score = func(trial)
                evals += 1
                
                if score < y[i]:
                    pop[i] = trial
                    y[i] = score
                    if score < best_y:
                        best_y = score
                        best_x = trial.copy()

        return best_x, best_y
