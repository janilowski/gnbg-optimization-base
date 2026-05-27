# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A standard (1+1)-CMA-ES style evolution strategy using isotropic mutation.
# Search state: Tracks current mean vector and adaptive step-size (sigma).
# Candidate generation: Generates a single offspring by sampling from a multivariate normal distribution around the current mean.
# Selection and replacement: Simple elitist selection: if the offspring is better than the current mean, it becomes the new mean.
# Adaptation: A simple 1/5th success rule is used to update the step-size sigma.
# Exploration mechanisms: Gaussian mutation with adaptive step-size controlled by the success rate.
# Exploitation mechanisms: Local hill climbing via progressive refinement of the mean and shrinking step-size.
# Boundary handling: Clipping candidates to the feasible region and reflecting them back into bounds for mutation.
# Budget strategy: Iterative loop that terminates exactly when the budget is exhausted.
# Closest known influences: (1+1)-ES with 1/5th success rule.
# Novelty or unusual aspects: Extremely compact implementation designed for black-box environments.
# Failure modes: Susceptible to local optima and premature convergence if the initial sigma is too small.
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
        
        # Initialize parameters
        x = np.random.uniform(lb, ub)
        y = func(x)
        evals = 1
        
        # Initial sigma as 1/5th of the range
        sigma = 0.2 * (ub - lb)
        successes = 0
        
        while evals < self.budget:
            # Generate offspring
            candidate = x + np.random.normal(0, sigma)
            # Clip and project
            candidate = np.clip(candidate, lb, ub)
            
            y_cand = func(candidate)
            evals += 1
            
            # Selection
            if y_cand <= y:
                x, y = candidate, y_cand
                successes += 1
            
            # 1/5th success rule adaptation every 10 iterations
            if evals % 10 == 0:
                if successes / 10 > 0.2:
                    sigma *= 1.2
                elif successes / 10 < 0.2:
                    sigma /= 1.2
                successes = 0
                
            if evals >= self.budget:
                break
                
        return x, y
