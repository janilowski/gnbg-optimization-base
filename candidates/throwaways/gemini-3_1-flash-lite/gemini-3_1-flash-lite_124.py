# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk (Hill Climbing with shrinkage).
# Search state: Current best point found.
# Candidate generation: Gaussian noise scaled by a step size added to the current best point.
# Selection and replacement: Greedy update; only move to the new point if its value is lower than current.
# Adaptation: Success-based step size control (1/5th rule inspired); shrinks when failing, expands on success.
# Exploration mechanisms: Initially broad search radius, becomes localized as performance improves.
# Exploitation mechanisms: Hill climbing behavior focused on improving the incumbent.
# Boundary handling: Points are clipped to valid bounds.
# Budget strategy: Pure sequential evaluation until the budget is exhausted.
# Closest known influences: Basic Stochastic Hill Climbing.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness and zero dependencies.
# Failure modes: Can get stuck in local minima; ineffective for extremely high-dimensional, rugged landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        curr_x = np.random.uniform(lb, ub)
        curr_y = func(curr_x)
        evals = 1

        best_x = np.copy(curr_x)
        best_y = curr_y

        # Adaptive step size (sigma)
        sigma = 0.2 * (ub - lb)
        
        while evals < self.budget:
            # Generate candidate using Gaussian perturbation
            candidate = curr_x + np.random.normal(0, sigma)
            # Boundary handling: clipping to ensure feasibility
            candidate = np.clip(candidate, lb, ub)
            
            y = func(candidate)
            evals += 1
            
            if y < curr_y:
                # Success: Move and expand search radius slightly
                curr_x = candidate
                curr_y = y
                sigma *= 1.1
                if y < best_y:
                    best_x = np.copy(candidate)
                    best_y = y
            else:
                # Failure: Shrink search radius to refine locally
                sigma *= 0.5
            
            # Reset sigma if it becomes too small to move
            if np.all(sigma < 1e-12):
                sigma = 0.1 * (ub - lb)
                
        return best_x, best_y
