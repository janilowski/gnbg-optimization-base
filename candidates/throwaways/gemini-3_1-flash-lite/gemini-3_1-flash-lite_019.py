# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Adaptive Random Search (ARS) with local mutation.
# Search state: Current best point found.
# Candidate generation: Gaussian noise added to current best, scaled by adaptive step size.
# Selection and replacement: Greedy update; if a candidate improves the best, move there.
# Adaptation: Step size increases after successful updates and decreases after failures.
# Exploration mechanisms: Initial large step sizes and Gaussian diffusion.
# Exploitation mechanisms: Step size decay and repeated local sampling near optima.
# Boundary handling: Clipping to provided bounds.
# Budget strategy: Iterative loop until function calls reach zero.
# Closest known influences: Simple adaptive random search/evolutionary strategy.
# Novelty or unusual aspects: Compact, single-loop control without population overhead.
# Failure modes: Slow convergence on highly needle-in-a-haystack (deceptive) landscapes.
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
        curr_x = np.random.uniform(lb, ub)
        curr_y = func(curr_x)
        remaining = self.budget - 1
        
        best_x = np.copy(curr_x)
        best_y = curr_y
        
        # Adaptive step size (sigma)
        sigma = 0.1 * (ub - lb)
        
        while remaining > 0:
            # Generate candidate
            candidate = curr_x + np.random.normal(0, sigma, self.dim)
            candidate = np.clip(candidate, lb, ub)
            
            y = func(candidate)
            remaining -= 1
            
            # Selection/Replacement
            if y < curr_y:
                curr_x = candidate
                curr_y = y
                # Expand step size on success
                sigma = np.minimum(sigma * 1.2, 0.5 * (ub - lb))
                
                if curr_y < best_y:
                    best_x = np.copy(curr_x)
                    best_y = curr_y
            else:
                # Shrink step size on failure (exploitation)
                sigma = np.maximum(sigma * 0.8, 1e-6 * (ub - lb))
                
            # Random restart if stuck
            if np.all(sigma < 1e-5 * (ub - lb)):
                sigma = (ub - lb) * 0.2
                curr_x = np.random.uniform(lb, ub)
                curr_y = func(curr_x)
                remaining -= 1
                
        return best_x, best_y
