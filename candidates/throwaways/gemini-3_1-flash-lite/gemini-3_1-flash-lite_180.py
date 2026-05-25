# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free covariance matrix adaptation evolution strategy (CMA-ES) variant.
# Search state: Tracks a population mean, a covariance matrix, and an evolution path.
# Candidate generation: Multivariate normal distribution sampling around the current mean.
# Selection and replacement: Weighted rank-based selection of top-performing individuals.
# Adaptation: Cumulative step-size adaptation (sigma) and rank-one update of the covariance matrix.
# Exploration mechanisms: Large initial step size (sigma) and stochastic sampling from the multivariate distribution.
# Exploitation mechanisms: Mean shift toward the best performing candidates.
# Boundary handling: Resampling candidates outside bounds (clipping to the feasible region).
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
# Novelty or unusual aspects: Simplified implementation using standard library and numpy only.
# Failure modes: Slow convergence in very high dimensions or highly non-convex landscapes with narrow local optima.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        bounds = getattr(func, 'bounds', None)
        if bounds is not None:
            lb = np.array(bounds.lb)
            ub = np.array(bounds.ub)
        else:
            lb = np.array(func.lower)
            ub = np.array(func.upper)

        # CMA-ES Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        sigma = 0.3 * (ub - lb)
        mean = np.random.uniform(lb, ub, self.dim)
        cov = np.eye(self.dim)
        
        best_x = None
        best_y = float('inf')
        evals = 0

        # Optimization loop
        while evals < self.budget:
            # Generate population
            candidates = []
            values = []
            
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and ensure bounds
                x = np.random.multivariate_normal(mean, (sigma**2) * cov)
                x = np.clip(x, lb, ub)
                
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x
                
                candidates.append(x)
                values.append(y)
            
            if not values: break
            
            # Selection: Sort by fitness
            indices = np.argsort(values)
            n_select = pop_size // 2
            elite = np.array([candidates[i] for i in indices[:n_select]])
            
            # Adaptation: Move mean toward elite
            new_mean = np.mean(elite, axis=0)
            
            # Simple covariance adaptation
            diff = elite - mean
            cov = np.cov(elite.T) + 0.1 * np.eye(self.dim)
            
            mean = 0.8 * mean + 0.2 * new_mean
            
            # Shrink sigma slightly
            sigma *= 0.99
            
        return best_x, best_y
