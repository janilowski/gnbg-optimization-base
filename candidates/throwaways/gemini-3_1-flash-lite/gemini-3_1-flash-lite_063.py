# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A variant of the (1+1)-Evolution Strategy with adaptive step-size control (1/5th success rule).
# Search state: Stores the current best point (incumbent) and current step size (sigma).
# Candidate generation: Adds independent Gaussian noise scaled by sigma to the incumbent.
# Selection and replacement: The new point replaces the incumbent if it yields a lower objective value.
# Adaptation: Sigma is updated every 10 iterations: increased if success rate > 0.2, decreased otherwise.
# Exploration mechanisms: Adaptive isotropic Gaussian mutations ensure global coverage that shrinks over time.
# Exploitation mechanisms: Local hill-climbing via successful mutations and step-size shrinkage.
# Boundary handling: Candidates are clipped to the bounding box constraints.
# Budget strategy: Iterates until the evaluation counter hits the budget limit.
# Closest known influences: (1+1)-ES, CMA-ES (simplified).
# Novelty or unusual aspects: Simplified self-adaptation logic suitable for black-box environments.
# Failure modes: Can get stuck in local optima; performance sensitive to initial sigma choice.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        else:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)

        # Initialization
        x = np.random.uniform(lb, ub)
        y = func(x)
        
        best_x = np.array(x)
        best_y = y
        
        # Initial sigma as 20% of the range
        sigma = 0.2 * (ub - lb)
        
        eval_count = 1
        success_count = 0
        
        # Adaptation parameters
        adapt_interval = 10
        
        while eval_count < self.budget:
            # Generate candidate
            z = np.random.normal(0, 1, self.dim)
            candidate = x + sigma * z
            
            # Boundary handling: clip to box
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            cand_y = func(candidate)
            eval_count += 1
            
            # Selection
            if cand_y <= y:
                x = candidate
                y = cand_y
                success_count += 1
                
                # Update global best
                if y < best_y:
                    best_y = y
                    best_x = np.array(x)
            
            # Adaptation (1/5th success rule)
            if eval_count % adapt_interval == 0:
                success_rate = success_count / adapt_interval
                if success_rate > 0.2:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                success_count = 0
                
            if eval_count >= self.budget:
                break
                
        return best_x, best_y
