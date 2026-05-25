# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the Differential Evolution (DE/rand/1/bin) strategy.
# Search state: Maintains a population of candidate solutions of size 10 + 2*dim.
# Candidate generation: Generates trial vectors using mutation (randomly selected individuals) and binomial crossover.
# Selection and replacement: Greedy selection; a trial vector replaces the target vector only if it provides a lower objective value.
# Adaptation: None; uses static control parameters (F=0.8, CR=0.9).
# Exploration mechanisms: Differential mutation (the difference between two random individuals scaled by F).
# Exploitation mechanisms: Greedy selection ensures the population converges towards local/global optima.
# Boundary handling: Reflective clipping to force candidates back into the provided hypercube.
# Budget strategy: Stops evaluation immediately once the budget is exhausted.
# Closest known influences: Storn and Price's Differential Evolution.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness in black-box continuous settings.
# Failure modes: May get trapped in local optima in highly multi-modal functions or struggle with high dimensions where F=0.8 is too aggressive.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(5, 10 + 2 * dim)
        self.f = 0.8
        self.cr = 0.9

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = func.lower, func.upper
        else:
            lb, ub = func.bounds.lb, func.bounds.ub
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Initialize population
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        evals = self.pop_size
        
        best_idx = np.argmin(fitness)
        best_x, best_y = pop[best_idx].copy(), fitness[best_idx]
        
        # Iterative evolution
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Mutation: pick 3 distinct individuals other than i
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Mutation and Crossover
                mutant = a + self.f * (b - c)
                mask = np.random.rand(self.dim) < self.cr
                # Ensure at least one index is updated
                mask[np.random.randint(0, self.dim)] = True
                
                trial = np.where(mask, mutant, pop[i])
                
                # Boundary handling: reflection
                trial = np.where(trial < lb, lb + (lb - trial), trial)
                trial = np.where(trial > ub, ub - (trial - ub), trial)
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                score = func(trial)
                evals += 1
                
                # Selection
                if score < fitness[i]:
                    fitness[i] = score
                    pop[i] = trial
                    if score < best_y:
                        best_y = score
                        best_x = trial.copy()
                        
        return best_x, best_y
