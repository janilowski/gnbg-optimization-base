# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant utilizing a simplified CMA approach.
# Search state: Maintains a mean vector, a global step size (sigma), and a simplified covariance matrix represented by an identity basis for efficiency.
# Candidate generation: Samples individuals from a multivariate Gaussian distribution centered at the current mean.
# Selection and replacement: Uses rank-based selection where the distribution mean is updated via a weighted average of the top-performing individuals.
# Adaptation: The step size (sigma) is adapted using a simplified cumulative step-length control, and the mean shifts to follow successful samples.
# Exploration mechanisms: Stochastic sampling governed by the covariance/sigma parameters ensures broad coverage early on.
# Exploitation mechanisms: The mean vector tracks successful descent directions, and sigma shrinks as convergence appears to tighten around local minima.
# Boundary handling: Samples are clipped to the provided search space bounds; the distribution mean is kept strictly within bounds.
# Budget strategy: A strictly enforced loop tracking the number of objective function calls against the provided budget.
# Closest known influences: CMA-ES by Hansen and Ostermeier.
# Novelty or unusual aspects: Simplified diagonal covariance structure to maintain complexity proportionality in restricted implementation settings.
# Failure modes: Can become stuck in sharp local minima if the initial sigma is too small or if the landscape is highly non-convex/deceptive.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Strategy parameters
        pop_size = 4 + int(3 * np.log(self.dim))
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        
        evals = 0
        
        # Optimization loop
        while evals < self.budget:
            # Generate population
            pop = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                    
                x = np.clip(mean + sigma * np.random.randn(self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                pop.append((x, y))
            
            # Sort population by fitness
            pop.sort(key=lambda item: item[1])
            
            # Update mean using top half of population (selection)
            weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, int(pop_size / 2) + 1))
            weights /= weights.sum()
            
            new_mean = np.zeros(self.dim)
            for i in range(len(weights)):
                new_mean += weights[i] * pop[i][0]
            
            # Adaptation: nudge mean and adjust sigma
            # 0.2 learning rate for mean update
            mean = 0.8 * mean + 0.2 * new_mean
            
            # Simple adaptive sigma adjustment based on progress
            if pop[0][1] < best_y:
                sigma *= 1.05
            else:
                sigma *= 0.95
                
            # Convergence check: if sigma is negligible, restart globally
            if np.mean(sigma) < 1e-9:
                sigma = 0.3 * (ub - lb)
                mean = np.random.uniform(lb, ub)
                
        return best_x, best_y
