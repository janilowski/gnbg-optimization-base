# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adapting Random Walk algorithm with a decaying step size (Simulated Annealing inspired).
# Search state: Current best point found so far and the current step size (sigma).
# Candidate generation: Gaussian perturbation of the current best point.
# Selection and replacement: Greedy replacement; only replaces if the new point is better.
# Adaptation: Sigma decreases over time (geometric decay) to focus on exploration initially and exploitation later.
# Exploration mechanisms: Initial large sigma values allow covering the search space.
# Exploitation mechanisms: Small sigma values refine the search around the current local optimum.
# Boundary handling: Clipping to the provided bounds.
# Budget strategy: Iterative step until the budget is exhausted.
# Closest known influences: Simple Random Walk, Simulated Annealing.
# Novelty or unusual aspects: Minimalist state management, robust to dimension scaling.
# Failure modes: Can get stuck in sharp local minima if the initial exploration is insufficient for the landscape.
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
        
        # Initialize
        budget_remaining = self.budget
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        budget_remaining -= 1
        
        # Parameters for adaptive walk
        # Start with a range covering 20% of the domain
        sigma = 0.2 * (ub - lb)
        
        while budget_remaining > 0:
            # Generate candidate using Gaussian perturbation
            candidate = best_x + np.random.normal(0, sigma)
            
            # Boundary handling: clip to bounds
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            current_y = func(candidate)
            budget_remaining -= 1
            
            # Selection and replacement
            if current_y < best_y:
                best_y = current_y
                best_x = candidate
                # Successive success: slightly broaden the step if needed (optional)
            else:
                # Decay the step size to focus exploitation
                sigma *= 0.999
                
            # Periodic reset if progress stalls (optional exploration boost)
            if budget_remaining % 1000 == 0:
                sigma = 0.05 * (ub - lb)
                
        return best_x, best_y
