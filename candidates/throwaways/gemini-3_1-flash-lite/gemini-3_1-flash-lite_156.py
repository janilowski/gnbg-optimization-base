# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE/rand/1/bin) global optimizer.
# Search state: Maintains a population of candidate vectors of size 10 + 2*dim.
# Candidate generation: Creates mutants using the difference of two random individuals scaled by a factor F=0.5.
# Selection and replacement: Standard tournament selection where a mutant replaces an individual if it improves the objective.
# Adaptation: None; parameters are fixed (F=0.5, CR=0.9).
# Exploration mechanisms: High population variance and differential mutation ensure global search.
# Exploitation mechanisms: Greedy replacement ensures the population contracts toward local minima.
# Boundary handling: Clamping to provided lower and upper bounds.
# Budget strategy: Distributed evenly, terminating once the population has evaluated the remaining budget.
# Closest known influences: Storn and Price's Differential Evolution.
# Novelty or unusual aspects: Minimalist implementation focused on reliability within constrained environments.
# Failure modes: Slow convergence on highly ill-conditioned or extremely high-dimensional landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(10, 2 * dim)
        self.max_gen = budget // self.pop_size

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        else:
            lb, ub = np.zeros(self.dim) - 5.0, np.zeros(self.dim) + 5.0

        # Initialize population
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        eval_count = self.pop_size

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # DE/rand/1/bin parameters
        F = 0.5
        CR = 0.9

        # Evolution loop
        for _ in range(self.max_gen):
            if eval_count >= self.budget:
                break
                
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                
                # Mutation: Select 3 distinct random individuals
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Recombination
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                score = func(trial)
                eval_count += 1
                
                if score <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = score
                    if score < best_y:
                        best_y = score
                        best_x = trial.copy()
        
        return best_x, best_y
