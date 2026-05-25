# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free covariance matrix adaptation evolution strategy (CMA-ES) variant utilizing a simplified rank-1 update.
# Search state: Maintains a mean vector (mu), a step size (sigma), and a simplified covariance proxy through an adaptive isotropic Gaussian distribution.
# Candidate generation: Samples population members from a multivariate normal distribution centered at the current mean.
# Selection and replacement: Selects the top-performing individuals to calculate a new weighted mean, shifting the distribution toward regions of lower cost.
# Adaptation: Updates the mean based on the best performing candidates; the step size evolves based on the progress of the mean over generations.
# Exploration mechanisms: Initialized with a large sigma covering the search space; stochastic sampling ensures coverage.
# Exploitation mechanisms: Mean-centering on successful points focuses the search locally once a promising basin is identified.
# Boundary handling: Clamping mechanism forces out-of-bounds candidates back to the nearest edge of the search space.
# Budget strategy: Divides budget into generations of a fixed population size until exhausted.
# Closest known influences: Simplified CMA-ES/Evolution Strategy (ES).
# Novelty or unusual aspects: Minimalist implementation of rank-based evolution without full matrix decomposition for compactness.
# Failure modes: May prematurely converge if the initial sigma is poorly scaled relative to the objective landscape.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        self.max_gens = budget // self.pop_size

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        evals = 0

        for gen in range(self.max_gens):
            # Generate population
            pop = [np.clip(mean + sigma * np.random.randn(self.dim), lb, ub) 
                   for _ in range(self.pop_size)]
            
            # Evaluate
            scores = []
            for x in pop:
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x
                scores.append(y)
                evals += 1
                if evals >= self.budget:
                    return best_x, best_y
            
            # Selection: Sort by performance and update mean
            indices = np.argsort(scores)
            weights = np.log(self.pop_size / 2 + 0.5) - np.log(np.arange(1, self.pop_size + 1))
            weights = np.maximum(0, weights / np.sum(weights))
            
            new_mean = np.zeros(self.dim)
            for i in range(self.pop_size):
                new_mean += weights[i] * pop[indices[i]]
            
            # Adaptation: Update step size based on movement
            diff = new_mean - mean
            mean = new_mean
            sigma *= np.exp(0.1 * (np.linalg.norm(diff) / (sigma + 1e-9) - 0.5))
            
        return best_x, best_y
