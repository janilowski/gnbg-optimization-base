# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant with a simplified diagonal covariance structure.
# Search state: Maintains a distribution center (mean) and a step-size (sigma).
# Candidate generation: Samples population members from a multivariate normal distribution centered at the mean.
# Selection and replacement: Selects the best performing fraction of the population to update the mean.
# Adaptation: Updates the mean toward the weighted average of the top performers; shrinks or expands step-size.
# Exploration mechanisms: Gaussian sampling ensures global coverage; step-size control prevents premature convergence.
# Exploitation mechanisms: Mean-shifting towards the current best-performing region.
# Boundary handling: Points sampled outside bounds are projected onto the nearest boundary.
# Budget strategy: Divides budget into generations; stops once budget is exhausted.
# Closest known influences: CMA-ES and (mu, lambda)-ES.
# Novelty or unusual aspects: Uses a diagonal covariance approximation for computational efficiency in high dimensions.
# Failure modes: May struggle with highly non-convex or needle-in-a-haystack landscapes; step-size can get trapped.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        self.mu = self.pop_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x_mean = np.random.uniform(lb, ub, self.dim)
        sigma = 0.3 * (ub - lb)
        best_x = np.copy(x_mean)
        best_y = float('inf')
        
        evals = 0
        
        while evals < self.budget:
            # Generate offspring
            population = []
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                x = np.clip(x_mean + sigma * np.random.randn(self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                population.append((x, y))
                
                if y < best_y:
                    best_y = y
                    best_x = x
            
            # Sort by fitness
            population.sort(key=lambda x: x[1])
            
            # Update mean
            old_mean = x_mean
            x_mean = np.sum([population[i][0] * self.weights[i] for i in range(self.mu)], axis=0)
            
            # Step size adaptation (simplified)
            sigma *= 0.95 + 0.1 * np.random.rand()
            
            # Convergence check
            if np.allclose(x_mean, old_mean, atol=1e-9):
                sigma *= 2.0
                
        return best_x, best_y
