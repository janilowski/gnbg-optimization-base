# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (Rastrigin/GNBG-friendly) Adaptive Random Search using a shrinking Gaussian mutation strategy.
# Search state: Tracks the current best solution (center) and a dynamic search radius (sigma).
# Candidate generation: Generates candidates via isotropic Gaussian noise added to the current best.
# Selection and replacement: Simple greedy update: if a candidate improves the best objective, it becomes the new center.
# Adaptation: Employs a 1/5th success rule heuristic to adapt the mutation step size (sigma).
# Exploration mechanisms: Initial large sigma allows global coverage; decay ensures local convergence.
# Exploitation mechanisms: Local hill-climbing via biased sampling around the incumbent.
# Boundary handling: Candidates are clipped to the search space using projection.
# Budget strategy: Iterative step-by-step evaluation until the budget is exhausted.
# Closest known influences: Adapted from (1+1)-ES (Evolution Strategy).
# Novelty or unusual aspects: Minimalist implementation using solely NumPy.
# Failure modes: Susceptible to local optima if the initial step size is too small; precision limits in high dimensions.
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
        current_x = np.random.uniform(lb, ub, self.dim)
        current_y = func(current_x)
        
        best_x = current_x.copy()
        best_y = current_y
        
        # Initial sigma as 1/5th of the range
        sigma = (ub - lb) * 0.2
        
        evals = 1
        success_count = 0
        
        while evals < self.budget:
            # Generate candidate
            candidate = current_x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            val = func(candidate)
            evals += 1
            
            # Update best
            if val < best_y:
                best_y = val
                best_x = candidate.copy()
            
            # Update center (greedy)
            if val < current_y:
                current_x = candidate
                current_y = val
                success_count += 1
            
            # Adapt sigma periodically (1/5th success rule)
            if evals % (self.dim * 2 + 1) == 0:
                success_ratio = success_count / (self.dim * 2 + 1)
                if success_ratio > 0.2:
                    sigma *= 1.2  # Too successful, expand search
                else:
                    sigma *= 0.8  # Not enough success, contract search
                success_count = 0
                
            # Convergence check: stop if sigma is effectively zero
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
