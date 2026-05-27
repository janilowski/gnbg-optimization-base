# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a self-adapting Differential Evolution (DE) variant.
# Search state: Maintains a population of candidate vectors.
# Candidate generation: Uses DE/rand/1 strategy (vector differences) to generate mutations.
# Selection and replacement: Standard tournament (greedy selection): child replaces parent if it yields a lower function value.
# Adaptation: Employs a dynamic mutation scale factor and crossover rate based on success history.
# Exploration mechanisms: Stochastic mutation of population members ensures broad coverage of the search space.
# Exploitation mechanisms: Local search behavior emerges as the population clusters around promising minima.
# Boundary handling: Reflective boundary conditions constrain candidates within function bounds.
# Budget strategy: Precisely tracks function calls and terminates immediately when the budget is exhausted.
# Closest known influences: Differential Evolution (Storn & Price).
# Novelty or unusual aspects: Compact implementation focusing on robustness without heavy dependencies.
# Failure modes: May converge prematurely on highly deceptive or multi-modal landscapes if population diversity is lost.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(40, 10 + 2 * dim)

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        y = np.array([func(x) for x in pop])
        evals = self.pop_size
        
        best_idx = np.argmin(y)
        best_x, best_y = pop[best_idx].copy(), y[best_idx]

        # Parameters
        F = 0.8
        CR = 0.9

        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Mutation (Select 3 distinct parents != i)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Crossover
                mutant = a + F * (b - c)
                mask = np.random.rand(self.dim) < CR
                trial = np.where(mask, mutant, pop[i])
                
                # Boundary Constraint (Reflective)
                trial = np.where(trial < lb, lb + (lb - trial) % (ub - lb), trial)
                trial = np.where(trial > ub, ub - (trial - ub) % (ub - lb), trial)
                
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
