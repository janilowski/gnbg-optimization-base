# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the Differential Evolution (DE/rand/1/bin) strategy.
# Search state: Maintains a population of candidate solutions.
# Candidate generation: Generates trial vectors using mutation (sum of random pair difference) and binomial crossover.
# Selection and replacement: Greedy selection; a new candidate replaces the target if it yields a lower function value.
# Adaptation: None; uses fixed hyper-parameters.
# Exploration mechanisms: Population diversity through random mutation and binomial crossover.
# Exploitation mechanisms: Greedy selection ensures the population converges towards local/global minima.
# Boundary handling: Clipping to the search space bounds.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: Storn and Price's Differential Evolution.
# Novelty or unusual aspects: Minimalist standard library implementation without external dependencies.
# Failure modes: May converge prematurely in high-dimensional or highly multi-modal landscapes with limited budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(max(10, 10 * dim), 200)
        self.F = 0.5  # Mutation factor
        self.CR = 0.9 # Crossover rate

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        else:
            lb, ub = -5.12, 5.12
            
        # Initialize population
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        budget_left = self.budget - self.pop_size
        
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]
        
        while budget_left > 0:
            for i in range(self.pop_size):
                if budget_left <= 0:
                    break
                
                # Mutation: pick 3 distinct random agents excluding i
                indices = [j for j in range(self.pop_size) if j != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                trial_fit = func(trial)
                budget_left -= 1
                
                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()
                        
        return best_x, best_y
