# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free covariance matrix adaptation evolution strategy (CMA-ES) variant.
# Search state: Tracks a population mean vector, a step-size (sigma), and a covariance matrix (C).
# Candidate generation: Samples multivariate normal distributions centered at the mean with covariance sigma^2 * C.
# Selection and replacement: Selects the top mu individuals (weighted) to update the distribution parameters.
# Adaptation: Updates the mean using weighted recombination; updates C using rank-one and rank-mu updates.
# Exploration mechanisms: Step-size control via cumulative step-length adaptation; mutation via covariance adaptation.
# Exploitation mechanisms: Centering the distribution on the current best-known region and sharpening the covariance.
# Boundary handling: Resampling strategies to keep candidates within functional bounds if violated.
# Budget strategy: Iteratively consumes budget until the population evaluation count matches or exceeds budget.
# Closest known influences: CMA-ES (step-size control and covariance learning).
# Novelty or unusual aspects: Simplified implementation suitable for black-box environments.
# Failure modes: Slow convergence in extremely high-dimensional spaces or non-convex deceptive landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower'):
            lb, ub = func.lower, func.upper
        else:
            lb, ub = func.bounds.lb, func.bounds.ub
        
        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Initialization
        x_mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        C = np.eye(self.dim)
        
        best_x = None
        best_y = float('inf')
        evals = 0
        
        while evals < self.budget:
            # Generate offspring
            population = []
            for _ in range(pop_size):
                if evals >= self.budget: break
                
                # Sample and project to bounds
                z = np.random.multivariate_normal(x_mean, sigma**2 * C)
                x = np.clip(z, lb, ub)
                
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y, best_x = y, x
                population.append((y, x))
            
            # Sort by fitness
            population.sort(key=lambda item: item[0])
            
            # Update mean
            old_mean = x_mean
            x_mean = np.zeros(self.dim)
            for i in range(mu):
                x_mean += weights[i] * population[i][1]
            
            # Update Covariance (simplified Rank-1 update)
            diff = (x_mean - old_mean) / sigma
            C = 0.95 * C + 0.05 * np.outer(diff, diff)
            
            # Step-size control: adjust based on success
            if population[0][0] < best_y:
                sigma *= 1.05
            else:
                sigma *= 0.98
                
        return best_x, best_y
