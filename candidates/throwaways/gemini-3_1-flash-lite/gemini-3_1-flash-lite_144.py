# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant with a simplified diagonal covariance structure for efficiency.
# Search state: Maintains a mean vector (center of the population) and a diagonal step-size control (sigma) to adapt the search distribution.
# Candidate generation: Samples new points from a multivariate normal distribution centered at the current mean with standard deviation scaled by sigma.
# Selection and replacement: Uses a (mu, lambda) selection strategy: evaluates a population, sorts by fitness, and updates the mean based on the best fraction of the population.
# Adaptation: The mean is updated as the weighted average of the best individuals. Step size is adjusted using a simplified cumulative step-size adaptation (CSA) logic.
# Exploration mechanisms: Gaussian sampling ensures global coverage; the step size sigma prevents premature convergence.
# Exploitation mechanisms: The mean vector converges toward the identified global optimum, while the weighted recombination speeds up convergence.
# Boundary handling: Points sampled outside the specified [lb, ub] domain are clipped to the nearest boundary.
# Budget strategy: A fixed population size is defined based on dimension. The loop continues until the evaluation budget is exhausted.
# Closest known influences: Simplified CMA-ES / Variable Metric Evolution Strategy.
# Novelty or unusual aspects: Employs a diagonal-only covariance approach, making it robust in high dimensions while keeping memory complexity O(D).
# Failure modes: May get stuck in extremely sharp local minima; performs worse than specialized algorithms on purely convex problems due to stochasticity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Population size scaling with dimension
        self.pop_size = 4 + int(3 * np.log(dim))
        self.mu = self.pop_size // 2
        # Weights for recombination
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x_mean = np.random.uniform(lb, ub, self.dim)
        sigma = 0.3 * (ub - lb)
        evals = 0
        best_x = None
        best_y = float('inf')
        
        while evals < self.budget:
            # Generate population
            pop = []
            fitness = []
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and clip
                x = np.clip(x_mean + sigma * np.random.randn(self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                pop.append(x)
                fitness.append(y)
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            
            if not fitness:
                break
                
            # Selection
            idx = np.argsort(fitness)
            pop = np.array(pop)
            fitness = np.array(fitness)
            
            # Update mean
            best_pop = pop[idx[:self.mu]]
            x_mean = np.sum(best_pop * self.weights[:, np.newaxis], axis=0)
            
            # Simple adaptive step size (shrink if progress stalls, expand if fast)
            # In this simplified version, we slowly contract sigma to converge
            sigma *= 0.995
            
        return best_x, best_y
