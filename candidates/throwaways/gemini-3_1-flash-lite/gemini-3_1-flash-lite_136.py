# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the Differential Evolution (DE/rand/1/bin) strategy.
# Search state: Maintains a population of candidate solutions of size 15 + 2 * dim.
# Candidate generation: Generates trial vectors using mutation (sum of random pair difference scaled by F=0.8) and binomial crossover (CR=0.9).
# Selection and replacement: Greedy selection; a trial vector replaces the target vector only if it yields a lower objective value.
# Adaptation: None; uses fixed standard DE parameters.
# Exploration mechanisms: Population-based stochastic mutation provides global exploration.
# Exploitation mechanisms: Greedy selection and recombination focus the search around current best regions.
# Boundary handling: Clamps candidates to the search space boundaries if they exceed them.
# Budget strategy: Calculates population size and number of generations based on the provided budget.
# Closest known influences: Storn and Price's 1997 Differential Evolution algorithm.
# Novelty or unusual aspects: Compact implementation suitable for standard library + numpy.
# Failure modes: May converge to local optima in highly deceptive landscapes; performance dependent on fixed parameters in very high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, 15 + 2 * dim)
        self.max_gens = (budget - self.pop_size) // self.pop_size
        self.F = 0.8  # Mutation scale factor
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = func.lower, func.upper
        else:
            lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize population
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]
        
        evals = self.pop_size
        
        # Evolution loop
        for _ in range(max(1, self.max_gens)):
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Mutation: pick 3 distinct individuals other than i
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Mutation and crossover
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                trial_f = func(trial)
                evals += 1
                
                if trial_f < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_y:
                        best_y = trial_f
                        best_x = trial.copy()
            
            if evals >= self.budget:
                break
                
        return best_x, best_y
