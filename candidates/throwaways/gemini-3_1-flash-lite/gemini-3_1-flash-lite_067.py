# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1)-Evolution Strategy with adaptive step-size control.
# Search state: Maintains a single candidate solution (x) and a global step-size (sigma).
# Candidate generation: Gaussian mutation N(0, sigma) added to current best.
# Selection and replacement: Greedy selection; if the mutant is better, it replaces the current solution.
# Adaptation: One-fifth success rule; sigma is scaled by 2.0 upon success, 1/1.5 upon failure.
# Exploration mechanisms: Gaussian mutation with adaptive step size ensures local search and diversity.
# Exploitation mechanisms: Hill-climbing behavior driven by the one-fifth success rule.
# Boundary handling: Simple projection (clamping) to function bounds.
# Budget strategy: Iteratively evaluates until remaining budget is exhausted.
# Closest known influences: (1+1)-ES, Rechenberg's 1/5th Rule.
# Novelty or unusual aspects: Extremely lightweight and robust for diverse objective landscapes.
# Failure modes: Can get trapped in local optima; performance sensitive to initial sigma choice for high dimensions.
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
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        remaining_budget = self.budget - 1
        
        # Adaptive parameters
        sigma = 0.2 * (ub - lb)
        successes = 0
        iterations = 0
        
        best_x, best_y = np.copy(x), y

        while remaining_budget > 0:
            # Generate mutant
            z = x + np.random.normal(0, sigma)
            z = np.clip(z, lb, ub)
            
            # Evaluate mutant
            y_z = func(z)
            remaining_budget -= 1
            iterations += 1
            
            # Selection
            if y_z < y:
                x, y = z, y_z
                successes += 1
                if y < best_y:
                    best_x, best_y = np.copy(x), y
            
            # One-fifth success rule adaptation every N iterations
            if iterations >= 10:
                ratio = successes / iterations
                if ratio > 0.2:
                    sigma *= 2.0
                elif ratio < 0.2:
                    sigma /= 1.5
                iterations = 0
                successes = 0
                
        return best_x, best_y
