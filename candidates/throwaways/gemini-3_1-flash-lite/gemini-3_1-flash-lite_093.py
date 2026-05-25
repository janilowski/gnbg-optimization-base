# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1)-CMA-ES variant using step-size adaptation (CSA).
# Search state: Current best point, step size (sigma), and coordinate-wise covariance matrix (identity).
# Candidate generation: Multivariate normal sampling centered at the best point with current sigma.
# Selection and replacement: Simple elitist (1+1) selection; updates best point if new value is lower.
# Adaptation: Sigma is increased after successful steps and decreased after failures (1/5th rule logic).
# Exploration mechanisms: Gaussian mutation scaled by adaptive step-size.
# Exploitation mechanisms: Elitist selection preserves the local optimum.
# Boundary handling: Simple projection (clipping) onto the feasible hypercube.
# Budget strategy: Deterministic loops until budget exhaustion.
# Closest known influences: (1+1)-ES with self-adaptive step-size.
# Novelty or unusual aspects: Minimalist implementation of adaptive covariance-like search.
# Failure modes: Can get trapped in narrow sharp local minima in very high dimensions.
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

        # Initialization
        x = np.random.uniform(lb, ub)
        y = func(x)
        
        # Hyperparameters for (1+1)-ES
        sigma = 0.2 * (ub - lb)
        success_rate = 0.0
        c_increase = 1.1
        c_decrease = 0.8
        
        best_x, best_y = np.copy(x), y
        eval_count = 1

        while eval_count < self.budget:
            # Generate candidate
            z = np.random.normal(0, 1, self.dim)
            candidate = best_x + sigma * z
            
            # Boundary handling
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            f_val = func(candidate)
            eval_count += 1
            
            # Selection
            if f_val < best_y:
                best_x, best_y = np.copy(candidate), f_val
                success_rate = 0.8 * success_rate + 0.2  # Success
                sigma *= c_increase
            else:
                success_rate = 0.8 * success_rate        # Failure
                sigma *= c_decrease
                
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 1e-10 * (ub - lb), 1.0 * (ub - lb))
            
            # Reset if stuck
            if eval_count % (self.dim * 10 + 10) == 0 and success_rate < 0.05:
                sigma = 0.1 * (ub - lb)

        return best_x, best_y
