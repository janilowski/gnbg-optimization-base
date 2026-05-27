# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (M)utative (S)tep-size adjustment.
# Search state: Tracks the current best solution and a global scalar step size (sigma).
# Candidate generation: Generates new candidates by adding Gaussian noise to the current best solution.
# Selection and replacement: Simple greedy acceptance; if a new candidate performs better, it replaces the current best.
# Adaptation: If a move is successful, the step size is increased (1.1x); if unsuccessful, it is decreased (0.5x).
# Exploration mechanisms: Gaussian sampling allows for wide-ranging search centered on best-known values.
# Exploitation mechanisms: Local hill-climbing via the adaptive step size shrinking around local optima.
# Boundary handling: Candidates are clipped to the provided search domain.
# Budget strategy: Uniformly partitions the budget into individual function evaluations, stopping exactly at budget completion.
# Closest known influences: Adaptive Step-Size Random Search, similar to a (1+1)-ES without recombination.
# Novelty or unusual aspects: Minimalist implementation using only standard library and basic numpy.
# Failure modes: Can get trapped in local optima; requires sufficient budget to navigate rugged landscapes.
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
        
        # Initialize
        curr_x = np.random.uniform(lb, ub, self.dim)
        curr_y = func(curr_x)
        self.budget -= 1
        
        best_x, best_y = curr_x.copy(), curr_y
        sigma = 0.2 * (ub - lb)  # Initial step size as percentage of range
        
        # Optimization loop
        while self.budget > 0:
            # Generate candidate via Gaussian mutation
            candidate = curr_x + np.random.normal(0, sigma, self.dim)
            candidate = np.clip(candidate, lb, ub)
            
            y = func(candidate)
            self.budget -= 1
            
            # Greedy comparison
            if y < curr_y:
                curr_x, curr_y = candidate.copy(), y
                # Update global best
                if y < best_y:
                    best_x, best_y = candidate.copy(), y
                # Expand step size for successful moves
                sigma = np.minimum(sigma * 1.1, (ub - lb))
            else:
                # Shrink step size for unsuccessful moves
                sigma = np.maximum(sigma * 0.5, 1e-8)
                
            # Periodic random restart if sigma is too small to escape flat regions
            if np.all(sigma < 1e-7):
                curr_x = np.random.uniform(lb, ub, self.dim)
                curr_y = func(curr_x)
                self.budget -= 1
                sigma = 0.1 * (ub - lb)
                
        return best_x, best_y
