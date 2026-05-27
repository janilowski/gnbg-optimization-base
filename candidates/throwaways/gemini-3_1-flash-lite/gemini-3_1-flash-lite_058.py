# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (1+1)-Evolution Strategy with adaptive step-size (self-adaptation).
# Search state: Current best point and current mutation step size (sigma).
# Candidate generation: Gaussian mutation of the current best point.
# Selection and replacement: Greedy selection; the offspring replaces the parent if it has a lower function value.
# Adaptation: The 1/5th success rule is used to increase sigma on success and decrease it on failure.
# Exploration mechanisms: Gaussian noise controlled by sigma.
# Exploitation mechanisms: Hill-climbing via the (1+1) selection mechanism.
# Boundary handling: Clipping to the search space bounds.
# Budget strategy: Iterates until the evaluation count reaches the budget.
# Closest known influences: Rechenberg's (1+1)-ES with 1/5th success rule.
# Novelty or unusual aspects: Extremely compact implementation for black-box global optimization.
# Failure modes: Can get stuck in local optima; requires sufficient budget for high dimensions.
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
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        evals = 1
        
        # Adaptive parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        while evals < self.budget:
            # Generate offspring
            x_new = np.clip(x + np.random.normal(0, sigma), lb, ub)
            y_new = func(x_new)
            evals += 1
            
            # Selection
            if y_new <= y:
                x, y = x_new, y_new
                success_count += 1
            else:
                success_count = 0
            
            # 1/5th Success Rule for step-size adaptation
            if evals % 5 == 0:
                if success_count / 5 > 0.2:
                    sigma *= 1.2  # Increase step size
                elif success_count / 5 < 0.2:
                    sigma /= 1.2  # Decrease step size
                success_count = 0
                
        return x, y
