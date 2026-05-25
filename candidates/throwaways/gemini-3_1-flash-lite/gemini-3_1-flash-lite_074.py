# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk with Shrunken Step Size.
# Search state: Tracks the current best-found point and its scalar objective value.
# Candidate generation: Generates candidates via normally distributed mutations (Gaussian perturbations).
# Selection and replacement: Greedy: a candidate replaces the best point only if its objective value is lower.
# Adaptation: Step size (sigma) decreases linearly over the budget to transition from exploration to fine-tuning.
# Exploration mechanisms: Initial large step sizes allow broad coverage of the search space.
# Exploitation mechanisms: Shrinking step sizes allow high-precision convergence near local minima.
# Boundary handling: Candidates are clipped to the hypercube defined by the objective function's bounds.
# Budget strategy: Iterative loop runs precisely until the budget is exhausted.
# Closest known influences: Stochastic Hill Climbing / (1+1)-ES with linear cooling.
# Novelty or unusual aspects: Minimalist implementation focusing on robust convergence within tight budgets.
# Failure modes: Can get trapped in deep local optima; performance is sub-optimal for highly deceptive landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'bounds'):
            lb, ub = func.bounds.lb, func.bounds.ub
        else:
            lb, ub = func.lower, func.upper
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Initialization: Random point within bounds
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        evals = 1
        
        # Initial sigma based on diagonal of the search space
        sigma = 0.2 * (ub - lb)
        
        while evals < self.budget:
            # Linear cooling schedule: reduce step size as budget approaches 0
            progress = evals / self.budget
            current_sigma = sigma * (1 - progress) + (sigma * 0.01) * progress
            
            # Candidate generation: Normal distribution perturbation
            candidate = best_x + np.random.normal(0, current_sigma, self.dim)
            
            # Boundary handling: Clip to defined limits
            candidate = np.clip(candidate, lb, ub)
            
            # Function evaluation
            y = func(candidate)
            evals += 1
            
            # Selection and replacement: Greedy update
            if y < best_y:
                best_y = y
                best_x = candidate
                
        return best_x, best_y
