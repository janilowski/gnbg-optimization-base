# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (GNBG) minimization implementation using a Simplified Covariance Matrix Adaptation Evolution Strategy (CMA-ES variant).
# Search state: Tracks the mean position (center of distribution) and a global step size (standard deviation).
# Candidate generation: Samples candidate vectors from a multivariate normal distribution centered at the current mean.
# Selection and replacement: Uses a rank-based selection: sorts the population by function values and updates the mean using the top-performing fraction of the population.
# Adaptation: The step size is adjusted via a simple 1/5th success rule logic, increasing if the best candidate improves upon the previous best.
# Exploration mechanisms: Gaussian noise provides high exploration; the step size controls the radius of the search in the parameter space.
# Exploitation mechanisms: Moving the mean towards the best candidates performs local gradient-free descent.
# Boundary handling: Candidates are clipped to the provided lower and upper bounds.
# Budget strategy: Divides the budget into generations, ensuring the total number of evaluations does not exceed the limit.
# Closest known influences: Inspired by CMA-ES and simple Evolution Strategies like (mu, lambda)-ES.
# Novelty or unusual aspects: Extremely compact implementation designed specifically for black-box environments with limited budgets.
# Failure modes: Can get trapped in sharp local optima or struggle on highly non-separable landscapes if the step size converges too quickly.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        num_parents = pop_size // 2
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = np.copy(mean)
        best_y = float('inf')
        
        evals = 0
        
        while evals < self.budget:
            # Generate population
            pop = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and clip
                x = np.clip(np.random.normal(mean, sigma), lb, ub)
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = np.copy(x)
                
                pop.append((x, y))
            
            if not pop:
                break
                
            # Selection: Sort by y
            pop.sort(key=lambda p: p[1])
            
            # Update mean: move towards top parents
            new_mean = np.mean([p[0] for p in pop[:num_parents]], axis=0)
            
            # Step size adaptation (1/5th success rule lite)
            if pop[0][1] < best_y:
                sigma *= 1.2
            else:
                sigma *= 0.95
                
            mean = new_mean
            
        return best_x, best_y
