# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple 1+1 Evolution Strategy (ES) with adaptive step size control.
# Search state: Current best individual (x) and scalar step size (sigma).
# Candidate generation: Gaussian mutation of the incumbent solution.
# Selection and replacement: Greedy selection; if child is better, replace parent.
# Adaptation: One-fifth success rule; increase sigma if success rate > 0.2, decrease otherwise.
# Exploration mechanisms: Gaussian mutation.
# Exploitation mechanisms: Local hill-climbing via adaptive step reduction.
# Boundary handling: Clipping to box constraints.
# Budget strategy: Iterative loop until evaluation count reaches budget.
# Closest known influences: Rechenberg's 1/5th success rule (1+1)-ES.
# Novelty or unusual aspects: Minimalist implementation focused on reliability.
# Failure modes: Can get trapped in local minima; limited by initial step size.
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
        evals = 1
        
        sigma = 0.2 * (ub - lb)
        successes = 0
        
        while evals < self.budget:
            # Generate candidate
            curr_sigma = np.diag(sigma)
            z = np.random.normal(0, 1, self.dim)
            x_new = np.clip(x + np.dot(sigma, z), lb, ub)
            
            y_new = func(x_new)
            evals += 1
            
            # Selection
            if y_new <= y:
                x, y = x_new, y_new
                successes += 1
            
            # Adaptation (1/5th success rule every 10 iterations)
            if evals % 10 == 0:
                if successes / 10 > 0.2:
                    sigma *= 1.2
                elif successes / 10 < 0.2:
                    sigma *= 0.8
                successes = 0
                
            if evals >= self.budget:
                break
                
        return x, y
