# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (C)entered (A)daptive (S)tep-sizes (RASCAS).
# Search state: Tracks the current best point (incumbent) and a dynamic standard deviation scaling factor.
# Candidate generation: Generates candidates via isotropic Gaussian noise added to the incumbent.
# Selection and replacement: Simple greedy update: replaces incumbent if the new point yields a lower value.
# Adaptation: Employs a 1/5th success rule to adapt the search radius; expands if frequent improvements occur, shrinks otherwise.
# Exploration mechanisms: Initial search is broad; Gaussian perturbation ensures global coverage.
# Exploitation mechanisms: Self-tuning step size allows local refinement around promising regions.
# Boundary handling: Uses clipping to keep candidates within the specified function optimization domain.
# Budget strategy: Iterates until the evaluation count matches the defined budget.
# Closest known influences: Evolutionary strategies and (1+1)-ES algorithms.
# Novelty or unusual aspects: Minimalist implementation of adaptive step-size without explicit covariance matrix update.
# Failure modes: Can get trapped in local basins or converge too slowly in high-dimensional narrow valleys.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds safely
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        curr_x = np.random.uniform(lb, ub)
        curr_y = func(curr_x)
        best_x, best_y = np.copy(curr_x), curr_y
        
        # Adaptive parameters
        sigma = 0.2 * (ub - lb)
        evals = 1
        success_count = 0
        
        # Main optimization loop
        while evals < self.budget:
            # Generate candidate using isotropic Gaussian perturbation
            candidate = curr_x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            val = func(candidate)
            evals += 1
            
            if val < curr_y:
                # Success: Move to new point and track improvement
                curr_x, curr_y = candidate, val
                success_count += 1
                if val < best_y:
                    best_x, best_y = np.copy(candidate), val
            
            # Periodically adapt sigma (every 10 evals or simple rolling adaptive factor)
            if evals % 10 == 0:
                # 1/5th success rule for step size control
                if success_count / 10 > 0.2:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                success_count = 0
                
        return best_x, best_y
