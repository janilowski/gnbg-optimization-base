# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adapting (1+1)-ES with covariance matrix adaptation (simplification of CMA-ES).
# Search state: A mean vector point x and a global step size sigma.
# Candidate generation: Gaussian perturbation of the mean: x_new = x + sigma * multivariate_normal(0, I).
# Selection and replacement: Greedy: update mean to x_new if f(x_new) < f(best).
# Adaptation: One-fifth success rule: increase sigma if success rate > 0.2, decrease if < 0.2.
# Exploration mechanisms: Global Gaussian sampling scaled by sigma.
# Exploitation mechanisms: Cumulative successful updates drive the mean toward the optimum.
# Boundary handling: Clipping candidates to the provided [lb, ub] range.
# Budget strategy: Iterative loop until evaluation count reaches budget.
# Closest known influences: (1+1)-Evolution Strategy with 1/5 success rule.
# Novelty or unusual aspects: Minimalist implementation of adaptive step size control.
# Failure modes: Getting trapped in needle-in-a-haystack traps or poor scaling if the landscape is highly non-convex.
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
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x = np.random.uniform(lb, ub)
        best_x = x.copy()
        best_y = func(x)
        
        # Adaptive strategy parameters
        sigma = 0.2 * (ub - lb)
        evals = 1
        success_count = 0
        
        # Iterative search
        while evals < self.budget:
            # Generate candidate
            z = np.random.normal(0, 1, self.dim)
            x_cand = np.clip(best_x + sigma * z, lb, ub)
            
            y_cand = func(x_cand)
            evals += 1
            
            # Selection
            if y_cand < best_y:
                best_y = y_cand
                best_x = x_cand.copy()
                success_count += 1
            
            # 1/5 success rule adaptation every 10 iterations
            if evals % 10 == 0:
                rate = success_count / 10
                if rate > 0.2:
                    sigma *= 1.2
                elif rate < 0.2:
                    sigma *= 0.8
                success_count = 0
                
            # Convergence check: shrink sigma if too small
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
