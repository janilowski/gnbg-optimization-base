# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk.
# Search state: Maintains the current best point found so far.
# Candidate generation: Generates candidates using a Gaussian perturbation around the best point, with step size scaled by a local adaptive factor.
# Selection and replacement: Greedy replacement—only replaces the best point if a candidate yields a lower objective value.
# Adaptation: The step size (sigma) increases after successful steps (to accelerate progress) and decreases after failed steps (to refine results).
# Exploration mechanisms: Initial large step sizes and Gaussian sampling provide broad coverage.
# Exploitation mechanisms: Shrinking step size allows for local refinement near minima.
# Boundary handling: Clamps candidates to the specified domain using numpy.clip.
# Budget strategy: A simple loop that terminates exactly when the evaluation count reaches the budget.
# Closest known influences: Simple (1+1)-ES (Evolution Strategy).
# Novelty or unusual aspects: Minimalist implementation using adaptive step-size control without complex covariance matrix updates.
# Failure modes: Can get trapped in local optima in highly multi-modal landscapes due to lack of global re-initialization.
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
        
        # Initialize state
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        
        evals = 1
        sigma = 0.2 * (ub - lb)  # Start with 20% of range
        
        while evals < self.budget:
            # Generate candidate using Gaussian perturbation
            candidate = best_x + np.random.normal(0, sigma, self.dim)
            
            # Boundary handling: clip to domain
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            current_y = func(candidate)
            evals += 1
            
            # Selection/Replacement
            if current_y < best_y:
                best_y = current_y
                best_x = candidate
                # Success: Expand search radius slightly
                sigma = np.minimum(sigma * 1.1, ub - lb)
            else:
                # Failure: Shrink search radius to refine
                sigma = np.maximum(sigma * 0.95, 1e-10)
                
            # Emergency break for very small step sizes
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
