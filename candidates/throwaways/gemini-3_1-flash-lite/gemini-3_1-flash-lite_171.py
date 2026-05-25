# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (C)entered (A)daptation.
# Search state: Tracks the current best-found point and a local search radius (scale).
# Candidate generation: Generates candidates via isotropic Gaussian mutation around the best point.
# Selection and replacement: Simple elitist replacement: only update the best point if a candidate is better.
# Adaptation: Adaptive step-size (1/5th rule) which shrinks the radius if mutations fail to improve the objective.
# Exploration mechanisms: Initial global search radius set to 1/4 of total bounds width.
# Exploitation mechanisms: Contractive noise reduction around the best-found candidate.
# Boundary handling: Simple projection (clipping) back into the valid domain.
# Budget strategy: Iterative loop that terminates exactly when the budget is exhausted.
# Closest known influences: Adaptive Random Search / (1+1) Evolution Strategy.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness to dimensionality.
# Failure modes: Can get stuck in local minima in highly rugged landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds safely
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        else:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)

        # Initialization
        x_best = np.random.uniform(lb, ub, self.dim)
        y_best = func(x_best)
        evals_done = 1
        
        # Initial sigma as 25% of the range
        sigma = 0.25 * (ub - lb)
        
        # Iterative search
        while evals_done < self.budget:
            # Generate mutation
            noise = np.random.normal(0, sigma)
            x_cand = np.clip(x_best + noise, lb, ub)
            
            y_cand = func(x_cand)
            evals_done += 1
            
            # Selection
            if y_cand < y_best:
                x_best = x_cand
                y_best = y_cand
                # Slightly expand search on success
                sigma = np.minimum(sigma * 1.1, (ub - lb))
            else:
                # Shrink search on failure (1/5th rule logic)
                sigma = np.maximum(sigma * 0.95, 1e-10)
                
            # If precision is too small, inject global noise to escape local minima
            if np.all(sigma < 1e-9):
                sigma = 0.1 * (ub - lb)
                
        return x_best, y_best
