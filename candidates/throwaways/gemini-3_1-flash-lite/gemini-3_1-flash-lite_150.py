# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (1+1)-Evolution Strategy with adaptive step-size control.
# Search state: Current candidate x and its objective value, plus adaptive step size sigma.
# Candidate generation: Gaussian mutation of the current best point.
# Selection and replacement: Greedy selection; new candidate replaces current if its value is lower.
# Adaptation: One-fifth success rule: increase sigma if success rate is high, decrease if low.
# Exploration mechanisms: Gaussian noise controlled by sigma.
# Exploitation mechanisms: Greedy search moves towards improved regions.
# Boundary handling: Simple clamping to the search space.
# Budget strategy: Iterates until budget is exhausted.
# Closest known influences: (1+1)-ES, CMA-ES (simplified).
# Novelty or unusual aspects: Extremely compact implementation for black-box environments.
# Failure modes: Can get trapped in local minima; step size might shrink too fast in high-dimensional landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower'):
            lb, ub = func.lower, func.upper
        else:
            lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialization
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        
        # Parameters for adaptation
        sigma = 0.2 * (ub - lb)
        success_count = 0
        evals = 1
        
        # Main optimization loop
        while evals < self.budget:
            # Generate candidate
            x_new = x + sigma * np.random.normal(0, 1, self.dim)
            x_new = np.clip(x_new, lb, ub)
            
            y_new = func(x_new)
            evals += 1
            
            # Selection
            if y_new <= y:
                x = x_new
                y = y_new
                success_count += 1
            
            # 1/5 Success Rule Adaptation (every 10 iterations)
            if evals % 10 == 0:
                success_rate = success_count / 10
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma /= 1.2
                success_count = 0
                
            if evals >= self.budget:
                break
                
        return x, y
