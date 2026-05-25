# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation strategy with a rank-1 update mechanism.
# Search state: Maintains a current mean vector and a step-size (sigma) for the multivariate normal distribution.
# Candidate generation: Generates a population of candidate points sampled from the current multivariate normal distribution.
# Selection and replacement: Uses a rank-based selection where the best candidate replaces the current mean.
# Adaptation: Updates the covariance-like step size based on success or failure of improvement; shrinks or expands to focus search.
# Exploration mechanisms: Initial search is broad (large sigma); stochastic sampling allows escaping local minima.
# Exploitation mechanisms: The mean consistently shifts towards successfully found minima.
# Boundary handling: Clipping candidates to the provided box constraints.
# Budget strategy: Iteratively consumes the budget using a population-based approach until depletion.
# Closest known influences: Simplified (1+1)-CMA-ES/Evolution Strategy.
# Novelty or unusual aspects: Compact implementation focusing on adaptive step-size without complex matrix inversion.
# Failure modes: Slow convergence in extremely high-dimensional spaces or highly deceptive/needle-in-a-haystack landscapes.
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
        x = np.random.uniform(lb, ub)
        y = func(x)
        remaining_budget = self.budget - 1
        
        # Adaptive step size (sigma)
        sigma = 0.2 * (ub - lb)
        best_x, best_y = x.copy(), y
        
        # Simple step-size adaptation (1/5th rule inspired)
        success_count = 0
        
        while remaining_budget > 0:
            # Generate candidate via mutation
            delta = np.random.normal(0, sigma)
            candidate = np.clip(x + delta, lb, ub)
            
            # Evaluate
            y_cand = func(candidate)
            remaining_budget -= 1
            
            # Selection/Replacement
            if y_cand < y:
                x = candidate
                y = y_cand
                success_count += 1
                if y < best_y:
                    best_x, best_y = x.copy(), y
            else:
                success_count -= 0.2 # Penalty for stagnation
            
            # Adaptation: adjust sigma every few iterations
            if (self.budget - remaining_budget) % 10 == 0:
                if success_count > 2:
                    sigma *= 1.2
                elif success_count < 0:
                    sigma *= 0.8
                success_count = 0
                
            # Convergence early exit: if sigma is tiny, refine with local hill climbing
            if np.all(sigma < 1e-9):
                sigma = 1e-4 * (ub - lb)

        return best_x, best_y
