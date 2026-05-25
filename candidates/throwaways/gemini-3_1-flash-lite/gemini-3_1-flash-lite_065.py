# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (1+1)-Evolution Strategy with adaptive step-size control.
# Search state: Maintains a single elite solution and a current mutation standard deviation (sigma).
# Candidate generation: Generates a new candidate by adding Gaussian noise to the parent.
# Selection and replacement: Uses a greedy selection; the candidate replaces the parent if its fitness is better.
# Adaptation: Employs the 1/5th success rule to increase sigma when success is frequent and decrease it when rare.
# Exploration mechanisms: Gaussian mutation with adaptive radius allows for global search in early stages and local refinement later.
# Exploitation mechanisms: The algorithm narrows the search space as it converges toward a local minimum.
# Boundary handling: Candidates are clipped to the provided search space bounds.
# Budget strategy: Iterates exactly until the evaluation budget is exhausted.
# Closest known influences: Basic (1+1)-ES, Rechenberg's 1/5th success rule.
# Novelty or unusual aspects: Minimalist implementation suitable for robust black-box benchmarking.
# Failure modes: Can get stuck in local optima; performance depends heavily on the initial sigma and convergence rate settings.
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
        
        # Initialization
        x = np.random.uniform(lb, ub)
        y = func(x)
        budget_left = self.budget - 1
        
        # Adaptive parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        total_count = 0
        
        while budget_left > 0:
            # Generate candidate
            z = np.random.normal(0, 1, self.dim)
            x_new = np.clip(x + sigma * z, lb, ub)
            
            y_new = func(x_new)
            budget_left -= 1
            
            # Selection
            if y_new <= y:
                x, y = x_new, y_new
                success_count += 1
            
            # Step size adaptation (1/5th success rule)
            total_count += 1
            if total_count >= 10:
                ratio = success_count / total_count
                if ratio > 0.2:
                    sigma *= 1.2
                elif ratio < 0.2:
                    sigma *= 0.8
                success_count = 0
                total_count = 0
                
        return x, y
