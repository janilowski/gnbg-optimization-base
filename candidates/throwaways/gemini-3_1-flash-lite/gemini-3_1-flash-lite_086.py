# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Step size (CMA-ES inspired heuristic).
# Search state: Tracks the current best solution and the current step size (sigma).
# Candidate generation: Generates candidates by sampling from a multivariate Normal distribution centered at the best solution.
# Selection and replacement: Greedy update; if a candidate improves the best solution, it becomes the new center.
# Adaptation: Sigma increases if a candidate improves the best result and decreases if repeated failures occur (1/5th success rule heuristic).
# Exploration mechanisms: Initial sigma is large (fraction of bounds); noise decays as the algorithm converges.
# Exploitation mechanisms: Local neighborhood search centered at current best.
# Boundary handling: Candidates are clipped to the defined search space bounds.
# Budget strategy: Uniform partition of the budget into a single opportunistic search phase.
# Closest known influences: (1+1)-ES (Evolution Strategy).
# Novelty or unusual aspects: Simplified self-adaptive step size adjustment without complex matrix decomposition.
# Failure modes: Can get stuck in local optima; ineffective for highly multimodal landscapes with distinct narrow basins.
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
        curr_x = np.random.uniform(lb, ub)
        curr_y = func(curr_x)
        
        best_x = np.copy(curr_x)
        best_y = curr_y
        
        # Initial sigma as 20% of the range
        sigma = 0.2 * (ub - lb)
        evals = 1
        
        # Adaptive parameters: 1/5th success rule
        success_count = 0
        
        while evals < self.budget:
            # Generate candidate
            candidate = curr_x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            y = func(candidate)
            evals += 1
            
            # Update best
            if y < best_y:
                best_y = y
                best_x = np.copy(candidate)
            
            # Update current state (greedy)
            if y < curr_y:
                curr_x = np.copy(candidate)
                curr_y = y
                success_count += 1
            else:
                success_count = 0
            
            # Adaptation logic (1/5th success rule)
            if evals % 10 == 0:
                if success_count > 2:
                    sigma *= 1.2  # Expand
                elif success_count == 0:
                    sigma *= 0.8  # Contract
            
            # Stop if range is negligible
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
