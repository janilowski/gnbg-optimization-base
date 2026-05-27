# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (A)daptive (S)tep size, similar to a simplified (1+1)-ES.
# Search state: Tracks the best point found so far and an adaptive step-size scale (sigma).
# Candidate generation: Generates a candidate by adding Gaussian noise scaled by sigma to the best point.
# Selection and replacement: Simple elitist replacement: if the candidate is better than the current best, it becomes the new best.
# Adaptation: If a candidate improves the objective, sigma is increased (1.1x) to accelerate progress; otherwise, it is decreased (0.5x) to refine the search.
# Exploration mechanisms: Initial sigma is set to 10% of the domain range; random Gaussian mutations.
# Exploitation mechanisms: Local contraction of the sampling ball when no improvements are found.
# Boundary handling: Candidates are clipped to the specified [lower, upper] box constraints.
# Budget strategy: Iterates until the evaluation count meets the provided budget.
# Closest known influences: (1+1)-Evolution Strategy with 1/5th success rule logic.
# Novelty or unusual aspects: Minimalist implementation of adaptive step-size without complex covariance matrix updates.
# Failure modes: Can get trapped in local optima in highly non-convex, high-dimensional landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        else:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)

        # Initialize
        budget_remaining = self.budget
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        budget_remaining -= 1
        
        # Initial step size is 10% of the domain range
        sigma = 0.1 * (ub - lb)
        
        while budget_remaining > 0:
            # Generate candidate mutation
            candidate_x = best_x + np.random.normal(0, sigma)
            # Boundary constraint enforcement
            candidate_x = np.clip(candidate_x, lb, ub)
            
            candidate_y = func(candidate_x)
            budget_remaining -= 1
            
            # Selection
            if candidate_y < best_y:
                best_x = candidate_x
                best_y = candidate_y
                # Success: expand search radius to encourage progress
                sigma *= 1.1
            else:
                # Failure: shrink search radius to perform local refinement
                sigma *= 0.5
            
            # Reset sigma if it becomes too small to move
            if np.all(sigma < 1e-12):
                sigma = 0.1 * (ub - lb)
                
        return best_x, best_y
