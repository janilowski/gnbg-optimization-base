# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (A)daptive (S)tep size (SA) algorithm.
# Search state: Maintains the current best solution and an adaptive step size (sigma).
# Candidate generation: Generates candidates via normally distributed perturbations around the current best.
# Selection and replacement: Simple elitist replacement; update occurs if a candidate improves the best objective.
# Adaptation: Step size doubles on success and halves on failure to navigate local landscapes.
# Exploration mechanisms: Initial search is broad; step size adapts based on success rate.
# Exploitation mechanisms: Local search centered on the best-found point.
# Boundary handling: Candidates are clipped to the specific bounds provided by the function.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: Adaptive Random Search (ARS).
# Novelty or unusual aspects: Minimalist state management for high portability.
# Failure modes: Can get trapped in sharp local minima if the basin of attraction is very narrow.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialize
        budget_left = self.budget
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        budget_left -= 1
        
        # Adaptive step size initialization
        sigma = (ub - lb) * 0.1
        
        while budget_left > 0:
            # Generate candidate
            step = np.random.normal(0, sigma, self.dim)
            candidate = np.clip(best_x + step, lb, ub)
            
            # Evaluate
            current_y = func(candidate)
            budget_left -= 1
            
            # Update best and adapt step size
            if current_y < best_y:
                best_x = candidate
                best_y = current_y
                sigma *= 1.1  # Expand if successful
            else:
                sigma *= 0.5  # Shrink if failed
                
            # Reset sigma if it becomes too small to explore
            if np.mean(sigma) < 1e-9:
                sigma = (ub - lb) * 0.05
                
        return best_x, best_y
