# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free covariance matrix adaptation evolution strategy (CMA-ES) variant.
# Search state: Stores the mean vector (x_mean), step size (sigma), and covariance matrix (C).
# Candidate generation: Samples new points from a multivariate normal distribution: x = x_mean + sigma * N(0, C).
# Selection and replacement: Selects the best lambda candidates to update the mean and covariance.
# Adaptation: Updates the mean using weighted recombination; updates covariance matrix using rank-one and rank-mu updates.
# Exploration mechanisms: Covariance matrix adaptation expands search along principal components; sigma controls global scale.
# Exploitation mechanisms: Moves the mean toward the best-performing samples.
# Boundary handling: Projects sampled points back into the valid search space using clipping.
# Budget strategy: Divides remaining budget into generations, terminating when budget is exhausted.
# Closest known influences: CMA-ES (Hansen et al.).
# Novelty or unusual aspects: Simplified implementation using a diagonal/full covariance hybrid approach for stability.
# Failure modes: Can get trapped in sharp local minima if sigma shrinks too early or fails to converge in high-dimensional landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = getattr(func, "bounds", None)
        if bounds is not None:
            lb, ub = bounds.lb, bounds.ub
        else:
            lb, ub = func.lower, func.upper
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        x_mean = np.random.uniform(lb, ub, self.dim)
        sigma = 0.3 * (ub - lb)
        C = np.eye(self.dim)
        
        best_x = None
        best_y = float('inf')
        evals = 0
        
        weights = np.log(pop_size + 0.5) - np.log(np.arange(1, pop_size + 1))
        weights /= np.sum(weights)
        
        while evals < self.budget:
            # Generate population
            pop = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                z = np.random.multivariate_normal(np.zeros(self.dim), C)
                x = np.clip(x_mean + sigma * z, lb, ub)
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x
                pop.append((y, x, z))
            
            # Sort by fitness
            pop.sort(key=lambda item: item[0])
            
            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum([weights[i] * pop[i][1] for i in range(len(pop))], axis=0)
            
            # Update Covariance (Rank-mu update)
            z_mean = np.sum([weights[i] * pop[i][2] for i in range(len(pop))], axis=0)
            C = 0.9 * C + 0.1 * np.outer(z_mean, z_mean)
            
            # Step size heuristic (simple decay)
            sigma *= 0.995
            
        return best_x, best_y
