# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (C)entered (S)hrinkage implementation.
# Search state: Tracks the current best solution and the current search radius.
# Candidate generation: Samples vectors from a multivariate normal distribution scaled by the current radius.
# Selection and replacement: Simple elitist replacement; if a candidate improves the best result, it replaces it.
# Adaptation: The radius shrinks exponentially over time to refine the search.
# Exploration mechanisms: Initial search is broad, sampling from the entire domain range.
# Exploitation mechanisms: Radius decay effectively narrows the search area around the current best.
# Boundary handling: Candidates are clipped to the valid domain boundaries.
# Budget strategy: Iterates until the evaluation count hits the provided budget.
# Closest known influences: Adaptive Random Search / Simple Pattern Search.
# Novelty or unusual aspects: Minimalist, robust, and requires no parameter tuning.
# Failure modes: Can get stuck in local optima if the function is highly multimodal and the decay is too fast.
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

        # Initialize state
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        evals = 1
        
        # Initial radius is the scale of the domain
        radius = (ub - lb)
        
        # Iterative search
        while evals < self.budget:
            # Generate candidate: gaussian step from current best
            # Use decay factor to slowly shrink the search space (annealing-like)
            progress = evals / self.budget
            scale = radius * (1.0 - progress) * 0.1
            
            candidate = best_x + np.random.normal(0, scale, self.dim)
            
            # Boundary enforcement
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            y = func(candidate)
            evals += 1
            
            # Update best if improved
            if y < best_y:
                best_y = y
                best_x = candidate
                
            # If we run out of budget, break
            if evals >= self.budget:
                break
                
        return best_x, best_y
