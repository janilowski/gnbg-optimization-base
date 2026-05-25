# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A modular implementation of the (1+1)-CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for derivative-free black-box optimization.
# Search state: Maintains current mean solution, step-size (sigma), and a rank-one update to the covariance matrix.
# Candidate generation: Samples new points from a multivariate normal distribution centered at current mean scaled by sigma.
# Selection and replacement: Simple greedy replacement (elitism); if the candidate is better than the current mean, it becomes the new mean.
# Adaptation: Updates sigma using the success rule (1/5th success rule principle) and updates the covariance matrix to capture the recent successful search direction.
# Exploration mechanisms: Stochastic mutation via multivariate normal sampling; sigma acts as the exploration radius.
# Exploitation mechanisms: Rank-one update on covariance matrix ensures the search learns the local geometry of the function.
# Boundary handling: Projects candidates back into the feasible region using a simple clamp (clipping).
# Budget strategy: Iterative loop that terminates exactly when the evaluation count matches the budget.
# Closest known influences: Hansen's (1+1)-CMA-ES.
# Novelty or unusual aspects: Minimalist implementation focused on parameter-free adaptation for general black-box contexts.
# Failure modes: Can get trapped in local optima for highly multi-modal functions; performance highly dependent on the initial sigma.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        else:
            lb, ub = -5.0 * np.ones(self.dim), 5.0 * np.ones(self.dim)

        # Initialization
        x = np.random.uniform(lb, ub)
        y = func(x)
        budget_left = self.budget - 1
        
        # State variables
        sigma = 0.3 * (ub - lb)
        C = np.eye(self.dim)
        p_c = np.zeros(self.dim)
        
        best_x, best_y = np.copy(x), y
        
        # Step size adaptation parameters
        cc = 4.0 / (self.dim + 4.0)
        c1 = 2.0 / ((self.dim + 1.3)**2 + 1.0)
        
        while budget_left > 0:
            # Generate candidate
            z = np.random.multivariate_normal(np.zeros(self.dim), C)
            candidate = x + sigma * z
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            y_next = func(candidate)
            budget_left -= 1
            
            # Selection
            if y_next < y:
                # Success: update mean and covariance
                x_old = np.copy(x)
                x = candidate
                y = y_next
                
                if y < best_y:
                    best_x, best_y = np.copy(x), y
                
                # Update evolution path and covariance
                z_step = (x - x_old) / sigma
                p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc)) * z_step
                C = (1 - c1) * C + c1 * np.outer(p_c, p_c)
                
                # Increase sigma
                sigma *= 1.1
            else:
                # Failure: decrease sigma
                sigma *= 0.5
            
            # Ensure C stays positive definite
            C = (C + C.T) / 2
            
        return best_x, best_y
