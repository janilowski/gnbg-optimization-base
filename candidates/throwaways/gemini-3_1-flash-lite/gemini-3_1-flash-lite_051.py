# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (C)entered (S)ampling (RSCS).
# Search state: Tracks the current best point found so far and its associated function value.
# Candidate generation: Generates candidates using a normal distribution centered on the best-found point, with a decaying step size (sigma).
# Selection and replacement: Updates the current best if a new sample results in a lower objective value.
# Adaptation: Sigma is adapted based on progress: if an improvement is found, the search radius is expanded; if a certain number of evaluations pass without improvement, it shrinks.
# Exploration mechanisms: Initial large variance allows global coverage; shrinking variance focuses on local refinement.
# Exploitation mechanisms: Greedy updates centered on the incumbent best position.
# Boundary handling: Candidates are clipped to the provided search space constraints.
# Budget strategy: Uniform distribution of evaluations over the total budget, terminating precisely at the limit.
# Closest known influences: Adaptive Random Search / (1+1)-Evolution Strategy.
# Novelty or unusual aspects: Minimalist implementation using only standard library and NumPy, robustness via coordinate clipping.
# Failure modes: Can get trapped in local optima in highly deceptive landscapes; performance degrades in extremely high-dimensional spaces.
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

        # Initial random guess
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        
        # Adaptive step size (sigma)
        sigma = 0.2 * (ub - lb)
        evals = 1
        
        # Track stagnation to adjust step size
        no_improve_count = 0
        
        while evals < self.budget:
            # Generate candidate
            candidate = best_x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            y = func(candidate)
            evals += 1
            
            # Selection
            if y < best_y:
                best_y = y
                best_x = candidate
                # Expand search sphere upon success
                sigma = np.minimum(sigma * 1.1, ub - lb)
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Shrink search sphere upon stagnation
                if no_improve_count > (self.dim * 5):
                    sigma *= 0.5
                    no_improve_count = 0
            
            # Terminate if sigma becomes negligible
            if np.all(sigma < 1e-10):
                break
                
        return best_x, best_y
