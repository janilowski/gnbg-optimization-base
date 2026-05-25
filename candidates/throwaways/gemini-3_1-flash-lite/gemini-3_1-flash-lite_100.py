# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES-inspired evolution strategy using a diagonal covariance matrix.
# Search state: Maintains a mean vector, a diagonal step-size (standard deviation) vector, and a current best candidate.
# Candidate generation: Generates candidates by sampling from a multivariate normal distribution centered at the mean with a diagonal covariance.
# Selection and replacement: Uses a (mu, lambda) selection strategy focusing on the best performing individuals from each generation.
# Adaptation: Updates the mean based on the weighted average of the best individuals and increases/decreases step sizes based on successful steps.
# Exploration mechanisms: Initializing with a large step-size relative to the domain bounds and using stochastic sampling.
# Exploitation mechanisms: The step size shrinks as the population clusters around the current best location.
# Boundary handling: Clamps candidates to the function-defined bounds if they exceed them.
# Budget strategy: Divides the budget into generations, accounting for population size, and terminates when the budget is exhausted.
# Closest known influences: Simplified CMA-ES / Diagonal Adaptation Evolution Strategy.
# Novelty or unusual aspects: Compact implementation focusing on simplicity and stability for constrained black-box optimization.
# Failure modes: May get trapped in very narrow local optima; convergence can be slow in extremely high-dimensional spaces or hyper-rugged landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = (ub - lb) / 4.0
        best_x = None
        best_y = float('inf')
        evals = 0

        # Weights for recombination
        mu = self.pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()

        while evals < self.budget:
            # Generate offspring
            candidates = []
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample
                z = np.random.normal(0, 1, self.dim)
                x = mean + sigma * z
                x = np.clip(x, lb, ub)
                
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                
                candidates.append((x, y, z))
            
            # Sort by fitness
            candidates.sort(key=lambda item: item[1])
            
            # Update mean using top mu candidates
            old_mean = mean.copy()
            mean = np.zeros(self.dim)
            for i in range(mu):
                mean += weights[i] * candidates[i][0]
            
            # Adapt sigma (simple diagonal adaptation)
            step = (mean - old_mean) / (sigma + 1e-9)
            sigma *= np.exp(0.1 * (step**2 - 1))
            
            # Decay local influence to ensure progress
            sigma = np.clip(sigma, 1e-6, (ub - lb) * 0.5)

        return best_x, best_y
