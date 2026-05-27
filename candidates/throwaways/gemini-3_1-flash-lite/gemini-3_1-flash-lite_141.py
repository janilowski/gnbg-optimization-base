# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk.
# Search state: Maintains a single 'best_x' and 'best_y' found so far in the budget.
# Candidate generation: Generates candidates by adding Gaussian noise to the current best_x, scaled by a step size.
# Selection and replacement: Greedy selection; if a candidate yields a lower objective value, it becomes the new best_x.
# Adaptation: The step size (sigma) follows a 1/5th success rule, increasing when successful and decreasing when unsuccessful.
# Exploration mechanisms: Initial global search is encouraged by a relatively high starting sigma; shrinkage allows local refinement.
# Exploitation mechanisms: Local hill-climbing via biased sampling around the current global best.
# Boundary handling: Candidates are clipped to the provided search space boundaries.
# Budget strategy: Uniformly consumes the entire budget in iterative steps.
# Closest known influences: Simple adaptive stochastic hill climbing.
# Novelty or unusual aspects: Extremely compact, dependency-free implementation using minimal memory.
# Failure modes: Can get stuck in local optima; performance is sensitive to the initial sigma scaling.
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
        
        # Initialize
        budget_left = self.budget
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        budget_left -= 1
        
        # Initial step size (1/10 of the range)
        sigma = 0.1 * (ub - lb)
        
        # Iterative search
        while budget_left > 0:
            # Generate candidate
            candidate = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
            
            # Evaluate
            y = func(candidate)
            budget_left -= 1
            
            # Selection
            if y < best_y:
                best_x, best_y = candidate, y
                # Success: increase step size to explore further
                sigma = np.minimum(sigma * 1.2, (ub - lb) * 0.5)
            else:
                # Failure: decrease step size to refine locally
                sigma = np.maximum(sigma * 0.8, (ub - lb) * 1e-6)
                
        return best_x, best_y
