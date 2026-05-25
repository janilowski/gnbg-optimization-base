# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Self-Adaptive Differential Evolution variant using a greedy strategy.
# Search state: Maintains a population of candidate vectors.
# Candidate generation: Randomly selects individuals to generate mutation vectors using the rand/1/bin strategy.
# Selection and replacement: Evolution strategy where offspring replace parents only if they improve fitness.
# Adaptation: Uses a simplified self-adaptive crossover rate and scaling factor.
# Exploration mechanisms: Population diversity through random mutation and recombination.
# Exploitation mechanisms: Greedy selection ensures monotonic improvement of the best-found solution.
# Boundary handling: Clamping mechanism to project out-of-bounds individuals onto the search space edges.
# Budget strategy: Iteratively evaluates population until the function evaluation budget is exhausted.
# Closest known influences: DE/rand/1/bin Differential Evolution.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness in black-box scenarios.
# Failure modes: Slow convergence on highly deceptive or extremely high-dimensional landscapes.
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
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        else:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)

        # Initialize population
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        evals = self.pop_size
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Hyperparameters
        f_scale = 0.8
        cr = 0.9

        # Evolutionary loop
        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break

                # Mutation: pick 3 random distinct individuals excluding current
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # Mutation and Crossover
                mutant = a + f_scale * (b - c)
                mask = np.random.rand(self.dim) < cr
                # Ensure at least one element is swapped
                mask[np.random.randint(0, self.dim)] = True
                
                trial = np.where(mask, mutant, pop[i])
                
                # Boundary handling
                trial = np.clip(trial, lb, ub)
                
                # Selection
                f_trial = func(trial)
                evals += 1
                
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()

        return best_x, best_y
