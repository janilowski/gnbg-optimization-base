# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Self-Adaptive Differential Evolution variant using a greedy local search strategy.
# Search state: Maintains a population of candidate solutions and the current global best.
# Candidate generation: Generates new candidates using mutation (DE/rand/1) and crossover.
# Selection and replacement: Greedy selection: new candidate replaces its parent if it yields a lower function value.
# Adaptation: Uses a simplified self-adaptive approach where crossover rate and mutation factor are jittered.
# Exploration mechanisms: Large initial population spread and random perturbation facilitate global search.
# Exploitation mechanisms: Greedy replacement and local crossover ensure convergence towards the best found regions.
# Boundary handling: Clipping strategy forces candidates back into the permitted domain.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: DE/rand/1/bin Differential Evolution algorithm.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness in black-box scenarios.
# Failure modes: May struggle with highly deceptive landscapes or extremely narrow needle-in-a-haystack global optima.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(40, 4 + int(3 * np.log(dim)))

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        evals = self.pop_size
        
        best_idx = np.argmin(fitness)
        best_x = np.copy(pop[best_idx])
        best_y = fitness[best_idx]
        
        # Hyperparameters
        F = 0.8
        CR = 0.9
        
        # Evolution loop
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Mutation: Select 3 distinct individuals excluding current i
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a1, a2, a3 = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Differential variant: DE/rand/1
                mutant = a1 + F * (a2 - a3)
                mutant = np.clip(mutant, lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                trial_y = func(trial)
                evals += 1
                
                if trial_y < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = np.copy(trial)
            
            # Simple adaptation for exploration
            F = 0.5 + 0.5 * np.random.rand()
            
        return best_x, best_y
