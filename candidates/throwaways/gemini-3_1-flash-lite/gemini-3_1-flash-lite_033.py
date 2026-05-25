# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free, population-based evolution strategy utilizing CMA-ES-like covariance matrix adaptation simplified for a compact implementation.
# Search state: Maintains a current mean vector, a global step size (sigma), and a diagonal covariance approximation.
# Candidate generation: Samples new points from a multivariate normal distribution centered on the mean.
# Selection and replacement: Uses a rank-based selection where the best fraction of the population updates the mean.
# Adaptation: Updates step size based on success rate (1/5th rule) and adjusts the mean towards better solutions.
# Exploration mechanisms: Multidimensional Gaussian sampling with adaptive step size.
# Exploitation mechanisms: Mean displacement towards superior samples and rank-based elitism.
# Boundary handling: Saturation (clipping) to the search space defined by the objective function.
# Budget strategy: Iterative generations until the evaluation budget is exhausted.
# Closest known influences: Simplified (1+lambda) or (mu, lambda)-ES with basic adaptation.
# Novelty or unusual aspects: Minimalist implementation of covariance adaptive search suitable for black-box constraints.
# Failure modes: Can converge to local optima in highly deceptive landscapes or struggle with extreme ill-conditioning.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.max_pop = 20  # Population size per generation

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x_mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = np.copy(x_mean)
        best_y = float('inf')
        evals = 0
        
        # Initial evaluation
        y = func(best_x)
        evals += 1
        best_y = y

        while evals < self.budget:
            # Generate population
            pop_size = min(self.max_pop, self.budget - evals)
            candidates = []
            values = []
            
            for _ in range(pop_size):
                # Sample and clip
                x = np.random.normal(x_mean, sigma)
                x = np.clip(x, lb, ub)
                
                y = func(x)
                evals += 1
                
                candidates.append(x)
                values.append(y)
                
                if y < best_y:
                    best_y = y
                    best_x = np.copy(x)
            
            # Selection: Sort by performance
            idx = np.argsort(values)
            mu = max(1, pop_size // 2)
            parents_x = [candidates[i] for i in idx[:mu]]
            
            # Adaptation: Move mean towards parents
            new_mean = np.mean(parents_x, axis=0)
            
            # Simple step-size adaptation (1/5th rule idea)
            if values[idx[0]] < best_y:
                sigma *= 1.1
            else:
                sigma *= 0.95
                
            x_mean = 0.8 * x_mean + 0.2 * new_mean
            
            if evals >= self.budget:
                break
                
        return best_x, best_y
