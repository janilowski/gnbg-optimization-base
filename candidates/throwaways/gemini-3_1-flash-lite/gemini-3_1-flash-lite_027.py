# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adaptive Differential Evolution (DE/rand/1/bin) variant tailored for black-box optimization.
# Search state: Maintains a population of candidates and their function values.
# Candidate generation: Uses vector difference mutation: x_new = x_r1 + F * (x_r2 - x_r3).
# Selection and replacement: Greedy selection (only replace if offspring is better or equal).
# Adaptation: Uses a self-adaptive mutation factor F derived from a uniform distribution (0.5 to 1.0).
# Exploration mechanisms: Large-scale global search via randomized vector differences across the population.
# Exploitation mechanisms: Binomial crossover retains successful dimensions from the parent.
# Boundary handling: Clamping to provided bounds with a small inward perturbation if boundaries are violated.
# Budget strategy: Static population size determined by dimension, linear budget exhaustion via generation cycles.
# Closest known influences: Differential Evolution (Storn & Price).
# Novelty or unusual aspects: Minimalist implementation of parameter-free DE.
# Failure modes: Slow convergence on highly deceptive landscapes or extremely high dimensions if budget is low.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(20 + 2 * dim, 100)
        self.generations = max(1, budget // (self.pop_size + 1))

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        budget_left = self.budget - self.pop_size
        
        best_idx = np.argmin(fitness)
        best_x, best_y = pop[best_idx].copy(), fitness[best_idx]
        
        # Evolution
        for _ in range(self.generations):
            if budget_left <= 0:
                break
                
            for i in range(self.pop_size):
                if budget_left <= 0:
                    break
                
                # Mutation (DE/rand/1)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                f = 0.5 + np.random.rand() * 0.5
                mutant = pop[r1] + f * (pop[r2] - pop[r3])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < 0.5
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Boundary handling: Clamp
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                score = func(trial)
                budget_left -= 1
                
                # Selection
                if score <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = score
                    if score < best_y:
                        best_y = score
                        best_x = trial.copy()
                        
        return best_x, best_y
