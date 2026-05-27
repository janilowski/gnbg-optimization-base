# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (C)ovariance (A)daptation (L)ike Gaussian sampling.
# Search state: Maintains a running best solution (x_best, y_best) and a dynamic step size (sigma).
# Candidate generation: Samples new points from a multivariate normal distribution centered at the current best.
# Selection and replacement: Greedy update; if a sampled point is better than the current best, it becomes the new center.
# Adaptation: Sigma is halved if no improvement is found after a fixed number of trials (patience) and doubled after a successful update.
# Exploration mechanisms: Initial global search is broad; localized by shrinking sigma as no better points are found.
# Exploitation mechanisms: Local hill-climbing via neighborhood sampling.
# Boundary handling: Simple projection (clipping) to the search space bounds.
# Budget strategy: Iterative loop that terminates exactly when the budget is exhausted.
# Closest known influences: Adaptive Random Search / (1+1)-ES.
# Novelty or unusual aspects: Minimalist implementation without complex matrix decompositions.
# Failure modes: Can get stuck in local optima; inefficient for extremely high-dimensional, highly non-convex surfaces.
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
        
        # Hyperparameters
        sigma = 0.2 * (ub - lb)  # Initial search radius
        patience = 10
        iters_since_improvement = 0
        
        evals = 1
        while evals < self.budget:
            # Generate candidate
            candidate = np.clip(best_x + np.random.normal(0, sigma, self.dim), lb, ub)
            
            # Evaluate
            y = func(candidate)
            evals += 1
            
            # Selection
            if y < best_y:
                best_y = y
                best_x = candidate
                sigma = np.minimum(sigma * 1.1, ub - lb)  # Accelerate
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1
                
            # Adaptation: reduce exploration radius if stuck
            if iters_since_improvement >= patience:
                sigma *= 0.5
                iters_since_improvement = 0
                
            # Termination check
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
