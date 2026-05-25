# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adapting (1+1)-ES with covariance matrix adaptation (CMA)-lite features.
# Search state: Current best point and a mutation step-size (sigma).
# Candidate generation: Gaussian mutation centered at current best.
# Selection and replacement: Success is defined by finding a lower objective value; the population is always size 1.
# Adaptation: The 1/5th success rule is used to adjust sigma dynamically.
# Exploration mechanisms: Initial global search via large sigma, controlled by adaptation.
# Exploitation mechanisms: Local hill-climbing triggered by successful evaluations.
# Boundary handling: Clipping candidates to the defined bounds.
# Budget strategy: Iterative generation until the budget counter is exhausted.
# Closest known influences: (1+1)-Evolution Strategy with 1/5th success rule.
# Novelty or unusual aspects: Minimalist implementation of adaptive step size without complex matrix operations.
# Failure modes: Can get stuck in local optima; performance degrades on highly rugged landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x = np.random.uniform(lb, ub)
        y = func(x)
        remaining_budget = self.budget - 1
        
        # Adaptive parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        total_steps = 0
        
        best_x, best_y = x.copy(), y
        
        while remaining_budget > 0:
            # Generate candidate
            z = np.random.normal(0, 1, self.dim)
            candidate = np.clip(x + sigma * z, lb, ub)
            
            # Evaluate
            f_val = func(candidate)
            remaining_budget -= 1
            total_steps += 1
            
            # Selection/Replacement
            if f_val < y:
                x, y = candidate.copy(), f_val
                success_count += 1
                if y < best_y:
                    best_x, best_y = x.copy(), y
            
            # Adaptation (1/5th success rule)
            if total_steps >= 10:
                success_rate = success_count / total_steps
                if success_rate > 0.2:
                    sigma *= 1.25
                elif success_rate < 0.2:
                    sigma *= 0.8
                
                # Reset stats
                total_steps = 0
                success_count = 0
                
                # Ensure sigma doesn't vanish
                sigma = np.maximum(sigma, 1e-10 * (ub - lb))
                
        return best_x, best_y
