# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (GNBG) minimization algorithm using a restart-based local search with shrinking adaptive step sizes.
# Search state: Maintains the current best point found so far and a local search radius.
# Candidate generation: Generates candidates via symmetric Gaussian perturbations around the current best point.
# Selection and replacement: Greedy selection; updates the best point if a candidate yields a lower objective value.
# Adaptation: The search radius (step size) shrinks when no improvement is found for a fixed number of trials (stagnation).
# Exploration mechanisms: Initial large-radius perturbations allow broad scanning; restarts reset the radius if progress halts.
# Exploitation mechanisms: Local refinement around the best-known point with gradually decreasing variance.
# Boundary handling: Clipping candidates to the provided [lb, ub] domain.
# Budget strategy: Iterative loop until the evaluation count reaches the defined budget.
# Closest known influences: Stochastic Hill Climbing / Pattern Search.
# Novelty or unusual aspects: Decoupled local search stages with restart triggers.
# Failure modes: Susceptible to local minima; may struggle with highly non-convex landscapes given the simple radial decay.
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
        
        # Initialization
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        eval_count = 1
        
        # Scaling parameters
        radius = 0.5 * (ub - lb)
        min_radius = 1e-7 * (ub - lb)
        stagnation_limit = max(10, self.dim * 2)
        
        while eval_count < self.budget:
            # Stagnation monitoring
            stagnation = 0
            
            while stagnation < stagnation_limit and eval_count < self.budget:
                # Generate candidate: Gaussian noise scaled by current radius
                candidate = best_x + np.random.normal(0, radius)
                candidate = np.clip(candidate, lb, ub)
                
                y = func(candidate)
                eval_count += 1
                
                if y < best_y:
                    best_x, best_y = candidate, y
                    stagnation = 0 
                else:
                    stagnation += 1
            
            # Decay radius if no progress found via local search
            radius *= 0.5
            
            # Restart if radius is too small to escape local trap
            if np.all(radius < min_radius):
                radius = 0.2 * (ub - lb)
                best_x = np.random.uniform(lb, ub)
                best_y = func(best_x)
                eval_count += 1
                
        return best_x, best_y
