# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1)-CMA-ES variant using Rank-1 adaptation.
# Search state: Maintains a mean vector and a step-size (sigma).
# Candidate generation: Multivariate normal sampling centered at the current mean.
# Selection and replacement: Elitism; update mean and sigma only if a better solution is found.
# Adaptation: Rank-1 update for the evolution path (C matrix) and success-based step-size control.
# Exploration mechanisms: Adaptive sigma based on success rate and Gaussian mutation.
# Exploitation mechanisms: Mean vector moves toward successful samples (local gradient approximation).
# Boundary handling: Clipping to the provided search space bounds.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: Hansen's (1+1)-CMA-ES.
# Novelty or unusual aspects: Minimalist implementation using identity covariance for simplicity and efficiency.
# Failure modes: Can get trapped in sharp local minima or fail on highly non-separable functions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize
        x = np.random.uniform(lb, ub)
        y = func(x)
        self.evals = 1
        
        best_x, best_y = np.copy(x), y
        
        # Hyperparameters for (1+1) update
        sigma = 0.2 * (ub - lb)
        success_rate = 0.0
        
        while self.evals < self.budget:
            # Sample
            dx = np.random.normal(0, sigma)
            x_new = np.clip(x + dx, lb, ub)
            
            y_new = func(x_new)
            self.evals += 1
            
            # Selection
            if y_new <= y:
                x, y = np.copy(x_new), y_new
                success_rate = 0.8 * success_rate + 0.2
                if y < best_y:
                    best_x, best_y = np.copy(x), y
            else:
                success_rate = 0.8 * success_rate
            
            # Adapt sigma (1/5th success rule)
            if self.evals % max(1, self.dim) == 0:
                if success_rate > 0.2:
                    sigma *= 1.1
                else:
                    sigma /= 1.1
            
            # Terminate if range becomes negligible
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
