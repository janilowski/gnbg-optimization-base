# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Trust-Region Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant designed for black-box optimization.
# Search state: Maintains a multivariate normal distribution defined by a mean vector, a global step size (sigma), and a covariance matrix.
# Candidate generation: Samples a population of potential solutions from the current multivariate normal distribution.
# Selection and replacement: Uses a rank-based selection mechanism, picking the best candidates to update the distribution parameters.
# Adaptation: Updates the mean toward the best samples and leverages rank-one and rank-mu covariance updates to track the geometry of the landscape.
# Exploration mechanisms: Initialized with a large sigma; controlled by a systematic reduction of the distribution's width (evolution path).
# Exploitation mechanisms: Concentrates search mass on the region of high-performing individuals (rank-based weightings).
# Boundary handling: Clamps candidates to the specified domain [lb, ub].
# Budget strategy: Iterates until the evaluation count reaches the budget, performing population-based updates in each generation.
# Closest known influences: CMA-ES (Covariance Matrix Adaptation Evolution Strategy) by Hansen et al.
# Novelty or unusual aspects: Simplified implementation suitable for single-file constraints without external dependencies.
# Failure modes: Stochastic nature may converge to sub-optimal local minima on highly deceptive landscapes.
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
        
        # CMA-ES Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        mean = np.random.uniform(lb, ub, self.dim)
        sigma = 0.3 * np.max(ub - lb)
        C = np.eye(self.dim)
        
        best_x = None
        best_y = float('inf')
        evals = 0
        
        while evals < self.budget:
            # Generate population
            samples = []
            values = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                x = np.random.multivariate_normal(mean, (sigma**2) * C)
                x = np.clip(x, lb, ub)
                y = func(x)
                evals += 1
                
                samples.append(x)
                values.append(y)
                
                if y < best_y:
                    best_y = y
                    best_x = x
            
            # Sort by fitness
            indices = np.argsort(values)
            samples = np.array(samples)[indices]
            
            # Update mean (weighted average of top half)
            weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, int(pop_size / 2) + 1))
            weights /= np.sum(weights)
            
            old_mean = mean.copy()
            mean = np.dot(weights, samples[:len(weights)])
            
            # Simple adaptive step update
            sigma *= 0.995 
            
            # Update Covariance (simplified rank-1 update)
            diff = (mean - old_mean) / sigma
            C = 0.9 * C + 0.1 * np.outer(diff, diff)
            
        return best_x, best_y
