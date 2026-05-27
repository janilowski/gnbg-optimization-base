# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (C)entered (S)ampling (RSCS).
# Search state: Tracks the current best point found so far and the step size (sigma).
# Candidate generation: Generates candidates via isotropic Gaussian noise added to the best point.
# Selection and replacement: Greedy selection; if a candidate performs better than the current best, it replaces the best point.
# Adaptation: Employs a one-fifth success rule rhythm: if a move is successful, step size doubles; otherwise, it halves.
# Exploration mechanisms: Large step sizes allow broad exploration of the search space.
# Exploitation mechanisms: Shrinking step sizes allow local refinement around local minima.
# Boundary handling: Candidates are clipped to the defined bounds of the function.
# Budget strategy: Uniformly consumes the budget, stopping exactly when the budget is reached.
# Closest known influences: Adaptive Random Search and standard (1+1)-ES.
# Novelty or unusual aspects: Minimalist implementation of adaptive step-size control without complex covariance matrices.
# Failure modes: May get stuck in sharp, deep local minima in high-dimensional multimodal landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = func.lower, func.upper
        else:
            lb, ub = func.bounds.lb, func.bounds.ub
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Initial guess (center of space) and evaluation
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        budget_left = self.budget - 1
        
        # Initial step size (10% of search space range)
        sigma = 0.1 * (ub - lb)
        
        while budget_left > 0:
            # Generate candidate
            candidate = best_x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            y = func(candidate)
            budget_left -= 1
            
            # Selection and Adaptation (1/5th rule logic)
            if y < best_y:
                best_y = y
                best_x = candidate
                # Successful move: expand search
                sigma = np.minimum(sigma * 1.1, (ub - lb))
            else:
                # Failed move: shrink search
                sigma = np.maximum(sigma * 0.9, 1e-10)
                
            if budget_left == 0:
                break
                
        return best_x, best_y
