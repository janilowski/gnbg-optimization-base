# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with adaptive step-size scaling (a simple hill-climber variant).
# Search state: Tracks the current best solution and the current standard deviation (step size) for proposal generation.
# Candidate generation: Gaussian perturbation centered on the current best.
# Selection and replacement: Greedy selection; if a candidate is better than the current best, replace the best and update the search.
# Adaptation: The step size shrinks if no improvement is found after a cooling-off period, and increases if success is frequent.
# Exploration mechanisms: Gaussian sampling allows for both local refinement and occasional long-range jumps.
# Exploitation mechanisms: Greedy updates ensure the search always moves toward lower function values.
# Boundary handling: Clamping to the provided search domain.
# Budget strategy: Uniform sequential sampling until the budget is exhausted.
# Closest known influences: Adapted from basic (1+1)-ES (Evolution Strategy).
# Novelty or unusual aspects: Extremely compact implementation with reactive step-size tuning.
# Failure modes: Susceptible to local minima in highly multimodal landscapes or high-dimensional "valleys".
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        remaining = self.budget - 1
        
        # Hyperparameters for step size adaptation
        scale = (ub - lb) * 0.1
        success_count = 0
        
        while remaining > 0:
            # Generate candidate
            candidate = np.random.normal(best_x, scale)
            candidate = np.clip(candidate, lb, ub)
            
            y = func(candidate)
            remaining -= 1
            
            # Greedy update
            if y < best_y:
                best_y = y
                best_x = candidate
                success_count += 1
                # Increase step size if improving often
                if success_count > 5:
                    scale = np.minimum(scale * 1.2, (ub - lb) * 0.5)
                    success_count = 0
            else:
                # Decrease step size if stagnant
                scale = np.maximum(scale * 0.95, (ub - lb) * 1e-4)
                success_count = 0
                
        return best_x, best_y
