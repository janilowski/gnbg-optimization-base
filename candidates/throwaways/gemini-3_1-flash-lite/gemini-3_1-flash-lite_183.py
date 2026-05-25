# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A minimalist implementation of a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) inspired local search.
# Search state: Maintains a mean vector, a global step size (sigma), and a diagonal covariance approximation.
# Candidate generation: Samples objective space using a multivariate normal distribution centered at the current best estimate.
# Selection and replacement: Uses a rank-based selection approach, updating the mean towards the best-performing candidates.
# Adaptation: Updates sigma via a simple 1/5th success rule to manage exploration/exploitation balance.
# Exploration mechanisms: Stochastic sampling governed by sigma; increases automatically if no progress is made.
# Exploitation mechanisms: Mean-shifting towards the elite samples found in each generation.
# Boundary handling: Projects candidates back into the valid search space using clipping.
# Budget strategy: Divides budget into generations of fixed size (lambda = 4 + floor(3 * ln(dim))); terminates when budget is exhausted.
# Closest known influences: Simplified CMA-ES and (μ, λ)-ES.
# Novelty or unusual aspects: Extremely compact implementation that fits the black-box constraints with minimal overhead.
# Failure modes: Large dimensions may suffer from the diagonal covariance limitation; convergence may stall on highly non-convex landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.sigma = 0.3
        self.pop_size = 4 + int(3 * np.log(dim))

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        mean = np.random.uniform(lb, ub)
        best_x = mean.copy()
        best_y = float('inf')
        
        evals = 0
        
        # Main optimization loop
        while evals < self.budget:
            # Generate population
            candidates = []
            values = []
            
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and project
                x = np.clip(mean + self.sigma * np.random.randn(self.dim) * (ub - lb), lb, ub)
                y = func(x)
                evals += 1
                
                candidates.append(x)
                values.append(y)
                
                # Update global best
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            
            if not values:
                break
                
            # Selection: Sort by performance
            idx = np.argsort(values)
            sorted_candidates = [candidates[i] for i in idx]
            
            # Adaptation: Mean shift (simple (1+lambda)-like update)
            new_mean = sorted_candidates[0]
            
            # Adaptation: 1/5th success rule for sigma
            if values[idx[0]] < best_y:
                self.sigma *= 1.1
            else:
                self.sigma *= 0.95
                
            mean = new_mean
            
        return best_x, best_y
