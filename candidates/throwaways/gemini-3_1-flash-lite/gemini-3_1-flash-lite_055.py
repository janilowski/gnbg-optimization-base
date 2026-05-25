# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adapting (1+1)-ES with covariance matrix adaptation (CMA-lite) for black-box minimization.
# Search state: Tracks current best solution, step size (sigma), and a diagonal covariance matrix.
# Candidate generation: Multivariate normal sampling centered on the current best.
# Selection and replacement: Greedy selection (comma-strategy); replace only if the new candidate is strictly better.
# Adaptation: One-fifth success rule for sigma; diagonal covariance updates move along successful evolutionary paths.
# Exploration mechanisms: Adaptive isotropic and directional noise scaled by sigma and covariance.
# Exploitation mechanisms: Gaussian hill-climbing centered on the best-found point.
# Boundary handling: Clipping candidates to the defined search space bounds.
# Budget strategy: Deterministic per-iteration decrement until the budget is exhausted.
# Closest known influences: (1+1)-CMA-ES, basic adaptive step-size search.
# Novelty or unusual aspects: Minimalist diagonal-only covariance tracking to save memory and complexity.
# Failure modes: Can stall in local minima if the landscape is highly non-convex or deceptive.
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

        # Initialization
        x = np.random.uniform(lb, ub)
        y = func(x)
        self.budget -= 1
        
        best_x = np.copy(x)
        best_y = y
        
        sigma = 0.2 * (ub - lb)
        diag_c = np.ones(self.dim)
        p_c = np.zeros(self.dim)
        
        # Hyperparameters
        c_c = 4 / (self.dim + 4)
        c_sigma = 0.3
        
        while self.budget > 0:
            # Generate candidate: sample along principle axes
            z = np.random.normal(0, 1, self.dim)
            dx = sigma * (z * np.sqrt(diag_c))
            x_new = np.clip(best_x + dx, lb, ub)
            
            y_new = func(x_new)
            self.budget -= 1
            
            # Selection/Replacement
            if y_new < best_y:
                # Success: update best and adapt covariance (evolution path)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (dx / sigma)
                diag_c = (1 - c_c) * diag_c + c_c * (p_c**2)
                
                best_x = np.copy(x_new)
                best_y = y_new
                sigma *= 1.22  # Increase step size
            else:
                # Failure: decrease step size
                sigma *= 0.82
                
            # Convergence check: reset sigma if it becomes too small
            if np.all(sigma < 1e-10):
                sigma = 0.1 * (ub - lb)
                
        return best_x, best_y
