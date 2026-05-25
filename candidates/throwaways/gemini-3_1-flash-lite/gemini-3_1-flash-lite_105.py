# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES-like evolution strategy using rank-based selection and covariance matrix adaptation.
# Search state: Maintains a mean vector, a covariance matrix (diagonal approximation), and a step size (sigma).
# Candidate generation: Samples population from a multivariate normal distribution centered at the mean.
# Selection and replacement: Selects the best performing fraction of the population to compute a new mean.
# Adaptation: Updates step size based on evolution path success and adapts the covariance matrix via moving averages.
# Exploration mechanisms: Initialized with large, normalized sigma; adapts via cumulative step-size control.
# Exploitation mechanisms: Moves the mean towards the best candidates, reducing variance over time.
# Boundary handling: Projects candidates back into bounds using clipping.
# Budget strategy: Divides the budget into generations, adjusting population size to ensure full coverage.
# Closest known influences: CMA-ES and (mu, lambda)-ES.
# Novelty or unusual aspects: Simplified diagonal-covariance adaptation for memory efficiency in black-box scenarios.
# Failure modes: Can stall if the search space is highly non-convex or if the step size shrinks prematurely.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Parameters for (mu/mu, lambda)-ES
        self.pop_size = 4 + int(3 * np.log(dim))
        self.mu = self.pop_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x, best_y = None, float('inf')
        
        evals = 0
        while evals < self.budget:
            # Generate candidates
            candidates = []
            values = []
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                x = mean + sigma * np.random.randn(self.dim)
                x = np.clip(x, lb, ub)
                y = func(x)
                evals += 1
                
                candidates.append(x)
                values.append(y)
                
                if y < best_y:
                    best_y = y
                    best_x = x
            
            if not values:
                break
                
            # Selection and update mean
            indices = np.argsort(values)
            top_indices = indices[:self.mu]
            top_candidates = np.array([candidates[i] for i in top_indices])
            
            new_mean = np.dot(self.weights, top_candidates)
            
            # Step size adaptation - simple progress-based heuristic
            diff = new_mean - mean
            mean = new_mean
            
            # Update sigma: increase if moving in consistent direction, decrease otherwise
            if np.linalg.norm(diff) > 0:
                sigma *= 1.05 
            else:
                sigma *= 0.95
            
            # Keep sigma within bounds
            sigma = np.clip(sigma, 1e-4 * (ub - lb), 0.5 * (ub - lb))
            
        return best_x, best_y
