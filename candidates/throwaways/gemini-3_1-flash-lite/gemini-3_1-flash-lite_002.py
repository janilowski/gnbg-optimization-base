# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (A)daptive (S)tep size (SAS).
# Search state: Maintains the best known vector (incumbent) and a dynamic step size (sigma).
# Candidate generation: Samples perturbations from a multivariate normal distribution scaled by sigma.
# Selection and replacement: Greedy selection; if a candidate improves the incumbent, it updates the position and expands sigma.
# Adaptation: The "1/5th success rule": sigma increases after successful moves to accelerate, and decreases after failures to refine.
# Exploration mechanisms: Large initial sigma and stochastic sampling across the search space.
# Exploitation mechanisms: Shrinking sigma allows for local refinement around the current best candidate.
# Boundary handling: Candidates are clipped to the defined bounds after generation.
# Budget strategy: Iterative loop that terminates exactly when the evaluation counter reaches the budget.
# Closest known influences: Adaptive Random Search and the (1+1)-ES (Evolution Strategy).
# Novelty or unusual aspects: Minimalist implementation of adaptive step-size control without full covariance matrix adaptation.
# Failure modes: Can get trapped in narrow local minima if the dimensionality is very high or the landscape is highly deceptive.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = getattr(func, 'lower', getattr(func.bounds, 'lb', -5.12))
        ub = getattr(func, 'upper', getattr(func.bounds, 'ub', 5.12))
        
        # Initialize
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        evals = 1
        
        best_x = x.copy()
        best_y = y
        
        # Initial step size (1/10th of search range)
        sigma = 0.1 * (ub - lb)
        
        while evals < self.budget:
            # Generate candidate
            candidate = np.clip(best_x + np.random.normal(0, sigma, self.dim), lb, ub)
            candidate_y = func(candidate)
            evals += 1
            
            # Selection/Replacement
            if candidate_y < best_y:
                best_y = candidate_y
                best_x = candidate.copy()
                # Expand step size to speed up convergence
                sigma *= 1.2
            else:
                # Shrink step size to refine locally
                sigma *= 0.8
            
            # Reset sigma if it becomes too small to move
            if np.all(sigma < 1e-12):
                sigma = 0.1 * (ub - lb)
                
        return best_x, best_y
