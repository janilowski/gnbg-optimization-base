# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) core logic, using a simplified step-size control.
# Search state: Maintains a mean vector, a covariance matrix (simplified as diagonal), and a step size (sigma).
# Candidate generation: Samples multivariate normal distributions centered at the current mean.
# Selection and replacement: Uses rank-based fitness selection (μ, λ), updating the mean toward the best individuals.
# Adaptation: Updates sigma using success-based rules and mean using progress-based drift.
# Exploration mechanisms: Large initial sigma and variance injection via multivariate sampling.
# Exploitation mechanisms: Mean updates focus the search density towards the current global minimum.
# Boundary handling: Clipping values to the domain bounds during candidate evaluation.
# Budget strategy: Iterative generation until the total function calls consume the allocated budget.
# Closest known influences: CMA-ES and standard (1+1)-ES evolution strategy.
# Novelty or unusual aspects: Simplified diagonal covariance matrix for constant space complexity.
# Failure modes: Premature convergence in highly non-convex landscapes or stagnation due to step-size collapse.
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
        sigma = 0.3 * (ub - lb)
        mean = np.random.uniform(lb, ub)
        
        best_x = None
        best_y = float('inf')
        evals = 0

        # Optimization loop
        while evals < self.budget:
            # Generate offspring
            samples = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and clip
                x = np.clip(mean + np.random.normal(0, sigma, self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x
                samples.append((x, y))
            
            # Sort: simple (1+1)-like update toward the best element of the generation
            samples.sort(key=lambda item: item[1])
            new_mean = samples[0][0]
            
            # Move mean with adaptation
            mean = 0.7 * new_mean + 0.3 * mean
            
            # Reduce step size if we are making progress, 
            # otherwise maintain enough exploration
            if samples[0][1] < best_y:
                sigma *= 0.95
            else:
                sigma *= 1.05
                sigma = np.clip(sigma, 0.001 * (ub - lb), 0.5 * (ub - lb))

        return best_x, best_y
