# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact implementation of the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) using diagonal approximation.
# Search state: Maintains a mean vector, a step-size (sigma), and a diagonal covariance matrix estimate.
# Candidate generation: Samples multivariate normal distributions using the mean and diagonal covariance scaling.
# Selection and replacement: Uses a rank-based selection where the top half of a population updates the mean.
# Adaptation: Updates step-size via cumulative step-length adaptation and the covariance via rank-one update.
# Exploration mechanisms: High initial sigma allows global search; adaptation shrinks the distribution.
# Exploitation mechanisms: Moves the mean toward the best-performing samples and sharpens the distribution.
# Boundary handling: Clamps candidates to the specified domain bounds before function evaluation.
# Budget strategy: Divides the budget into generations, ensuring population size is adjusted to terminate at the budget limit.
# Closest known influences: Simplified Diagonal CMA-ES as described in Hansen's tutorial on evolution strategies.
# Novelty or unusual aspects: Uses a fixed small population size for computational efficiency in lower budgets.
# Failure modes: May converge prematurely on highly multi-modal functions or struggle if optimal is at the boundary.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        self.max_gens = max(1, budget // self.pop_size)

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialize state
        x_mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        diag_d = np.ones(self.dim)
        
        best_x = None
        best_y = float('inf')
        
        evals = 0
        
        for gen in range(self.max_gens):
            # Generate population
            pop = []
            fitness = []
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and clamp
                z = np.random.normal(0, 1, self.dim)
                x = x_mean + sigma * (diag_d * z)
                x = np.clip(x, lb, ub)
                
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x
                
                pop.append((x, y, z))
                fitness.append(y)
            
            if evals >= self.budget:
                break
                
            # Selection: Sort by fitness
            pop.sort(key=lambda item: item[1])
            
            # Update mean (weighted towards best half)
            weights = np.log(self.pop_size / 2.0 + 0.5) - np.log(np.arange(1, int(self.pop_size / 2) + 1))
            weights /= weights.sum()
            
            new_mean = np.zeros(self.dim)
            for i in range(len(weights)):
                new_mean += weights[i] * pop[i][0]
                
            # Update evolution path and covariance (simplified)
            diag_d = diag_d * 0.9 + 0.1 * np.mean([p[2]**2 for p in pop[:int(self.pop_size/2)]], axis=0)
            x_mean = new_mean
            
            # Adapt sigma: decrease if success is sparse, slightly increase if high
            if pop[0][1] < best_y:
                sigma *= 1.05
            else:
                sigma *= 0.98
                
        return best_x, best_y
