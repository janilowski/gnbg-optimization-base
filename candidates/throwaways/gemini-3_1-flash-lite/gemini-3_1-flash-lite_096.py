# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1)-CMA-ES variant using an isotropic mutation strategy.
# Search state: Maintains a current candidate (mean) and a global step size (sigma).
# Candidate generation: Samples a new candidate by adding Gaussian noise scaled by sigma to the current mean.
# Selection and replacement: Deterministic selection; the new candidate replaces the current mean if its function value is lower.
# Adaptation: Employs the 1/5th success rule to adapt the step size sigma.
# Exploration mechanisms: Gaussian mutation with adaptive sigma allows global coverage that shrinks over time.
# Exploitation mechanisms: Local hill-climbing via constant evaluation of the neighborhood.
# Boundary handling: Candidates are clipped to the provided lower and upper bounds.
# Budget strategy: A strict counter ensures the loop terminates exactly at the budget limit.
# Closest known influences: (1+1)-Evolution Strategy with 1/5th success rule.
# Novelty or unusual aspects: Minimalist implementation of CMA-style step-size control.
# Failure modes: Can get stuck in local optima; performance depends heavily on the initial sigma.
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
        
        # Initialize state
        curr_x = np.random.uniform(lb, ub, self.dim)
        curr_y = func(curr_x)
        
        best_x, best_y = np.copy(curr_x), curr_y
        
        sigma = 0.2 * (ub - lb)
        evals = 1
        success_count = 0
        
        while evals < self.budget:
            # Generate candidate
            candidate = curr_x + np.random.normal(0, sigma, self.dim)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            y = func(candidate)
            evals += 1
            
            # Selection
            if y < curr_y:
                curr_x, curr_y = candidate, y
                success_count += 1
                if y < best_y:
                    best_x, best_y = np.copy(candidate), y
            
            # Adapt sigma every 5 evaluations (1/5th rule)
            if evals % 5 == 0:
                if success_count / 5 > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                success_count = 0
                
        return best_x, best_y
