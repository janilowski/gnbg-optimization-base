# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple yet effective Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant limited to Diagonal Covariance (Var-CMA).
# Search state: Maintains a mean vector and a diagonal variance vector (step size).
# Candidate generation: Multivariate normal sampling centered at mean with diagonal covariance.
# Selection and replacement: Weighted rank-based selection using the best mu out of lambda offspring.
# Adaptation: Updates mean towards successful points and adapts step sizes (variance) based on rank-based progress.
# Exploration mechanisms: Initial large variance (1/3 of search space) and stochastic sampling.
# Exploitation mechanisms: Cumulative step-size adaptation and mean shift.
# Boundary handling: Simple clipping to target bounds.
# Budget strategy: Iteratively evaluates populations of size 4 + floor(3 * log(dim)) until budget exhaustion.
# Closest known influences: CMA-ES, specifically the diagonal-only implementation.
# Novelty or unusual aspects: Minimized memory footprint suitable for smaller budgets.
# Failure modes: Can converge to local minima in highly deceptive terrain.
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
        
        range_ = ub - lb
        
        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * range_  # Initial step size
        
        best_x = None
        best_y = float('inf')
        eval_count = 0
        
        while eval_count < self.budget:
            # Generate Candidate Population
            pop = []
            for _ in range(pop_size):
                if eval_count >= self.budget:
                    break
                
                # Sample and clip
                x = np.clip(mean + sigma * np.random.randn(self.dim), lb, ub)
                y = func(x)
                eval_count += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x
                pop.append((x, y))
            
            # Selection
            pop.sort(key=lambda x: x[1])
            best_mu = [p[0] for p in pop[:mu]]
            
            # Adaptation: Update mean
            old_mean = mean.copy()
            mean = np.dot(weights, best_mu)
            
            # Adaptation: Update step size (simplified path length control)
            z = (mean - old_mean) / (sigma + 1e-9)
            sigma *= np.exp(np.clip((np.sum(z**2) / self.dim) - 0.25, -0.2, 0.2))
            
        return best_x, best_y
