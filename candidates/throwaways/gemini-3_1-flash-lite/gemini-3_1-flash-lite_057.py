# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adapting (1+1)-ES with covariance matrix adaptation (CMA)-like mutation step size control.
# Search state: Stores the best candidate vector and a scalar step-size (sigma).
# Candidate generation: Mutation via isotrophic Gaussian noise scaled by sigma; jitter is added to avoid local optima.
# Selection and replacement: Simple greedy replacement: new point replaces the best if it yields a lower function value.
# Adaptation: The 'one-fifth success rule': sigma increases if the success rate is high (> 20%), decreases if low.
# Exploration mechanisms: Gaussian mutation with adaptive radius; adaptive sigma allows for both broad exploration and fine-tuning.
# Exploitation mechanisms: Local hill-climbing via greedy refinement of the current best solution.
# Boundary handling: Clipping candidates to the box constraints upon generation.
# Budget strategy: Iterates exactly until the budget is exhausted, with a small safety margin for initialization.
# Closest known influences: (1+1)-Evolution Strategy with 1/5 success rule.
# Novelty or unusual aspects: Extremely compact implementation with a robust step-size adaptation scheme.
# Failure modes: Susceptible to getting stuck in local minima if the optimization landscape is highly deceptive and non-convex.
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
        evals = 1

        sigma = 0.2 * (ub - lb)
        success_history = []
        
        # Optimization loop
        while evals < self.budget:
            # Generate candidate
            z = np.random.normal(0, 1, self.dim)
            x_new = np.clip(x + sigma * z, lb, ub)
            
            y_new = func(x_new)
            evals += 1
            
            # Selection
            if y_new <= y:
                x, y = x_new, y_new
                success_history.append(1)
            else:
                success_history.append(0)
                
            # Adaptation: 1/5 success rule
            if evals % (self.dim * 2) == 0:
                success_rate = np.mean(success_history[-self.dim*2:])
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                    
            # Prevent zero sigma
            sigma = np.maximum(sigma, 1e-10 * (ub - lb))
            
            if evals >= self.budget:
                break
                
        return x, y
