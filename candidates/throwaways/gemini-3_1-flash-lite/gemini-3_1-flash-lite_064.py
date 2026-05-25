# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (GNBG) trust-region-inspired random search.
# Search state: Tracks a single best solution (incumbent) and a dynamic search radius.
# Candidate generation: Sampling from a multivariate normal distribution centered on the incumbent.
# Selection and replacement: Greedy replacement if a candidate improves the objective.
# Adaptation: Radius shrinks (exploitation) if no improvement is found, grows (exploration) if success is frequent.
# Exploration mechanisms: Adaptive radius ensures a mix of local refinement and global coverage.
# Exploitation mechanisms: Local improvement by concentrating samples around the current best.
# Boundary handling: Clipping candidates to the feasible domain.
# Budget strategy: Uniform linear consumption until the budget is exhausted.
# Closest known influences: Adaptive Random Search / Trust-region methods.
# Novelty or unusual aspects: Extremely compact 1/5th success rule implementation.
# Failure modes: Slow convergence on highly needle-in-a-haystack problems or very high dimensions.
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
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        
        best_x = x.copy()
        best_y = y
        
        # Search parameters
        radius = 0.2 * (ub - lb)
        evals = 1
        success_count = 0
        
        while evals < self.budget:
            # Generate candidate
            step = np.random.normal(0, radius)
            candidate = np.clip(best_x + step, lb, ub)
            
            cand_y = func(candidate)
            evals += 1
            
            # Selection
            if cand_y < best_y:
                best_x = candidate
                best_y = cand_y
                success_count += 1
            
            # Adaptation (1/5th success rule logic)
            if evals % (self.dim + 1) == 0:
                if success_count / (self.dim + 1) > 0.2:
                    radius *= 1.2
                elif success_count / (self.dim + 1) < 0.2:
                    radius *= 0.8
                success_count = 0
                
        return best_x, best_y
