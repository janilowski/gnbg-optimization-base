# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk (Hill Climbing with shrinkage).
# Search state: Current best point found.
# Candidate generation: Gaussian noise scaled by a step size added to the current best.
# Selection and replacement: Greedy acceptance (only updates if a lower objective value is found).
# Adaptation: The step size 'sigma' is halved if no improvement is found after a few iterations, allowing refinement.
# Exploration mechanisms: Initial large step size for global coverage, shrinking over time.
# Exploitation mechanisms: Local hill climbing following the negative gradient vector via small perturbations.
# Boundary handling: Clamping candidate points to the provided box constraints.
# Budget strategy: Iterative loop until internal budget counter reaches zero.
# Closest known influences: Adaptive Step Size Random Search / Hill Climbing.
# Novelty or unusual aspects: Minimalist state management for memory-constrained black-box scenarios.
# Failure modes: Can get trapped in sharp local minima if the global minimum is far from the initial point.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'bounds'):
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        else:
            lb, ub = np.array(func.lower), np.array(func.upper)

        # Initialize: Start at center of space or random
        current_x = np.random.uniform(lb, ub)
        current_y = func(current_x)
        self.budget -= 1
        
        best_x = current_x.copy()
        best_y = current_y
        
        # Hyperparameters
        sigma = (ub - lb) * 0.2  # Initial search radius
        stagnation_limit = 10    # Patience before shrinking step size
        stagnation_counter = 0

        while self.budget > 0:
            # Generate candidate
            candidate = current_x + np.random.normal(0, sigma, size=self.dim)
            
            # Boundary enforcement
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluation
            y = func(candidate)
            self.budget -= 1
            
            # Selection
            if y < current_y:
                current_x = candidate.copy()
                current_y = y
                stagnation_counter = 0
                
                # Update global best
                if y < best_y:
                    best_x = candidate.copy()
                    best_y = y
            else:
                stagnation_counter += 1
            
            # Adaptation: Reduce step size if stuck
            if stagnation_counter >= stagnation_limit:
                sigma *= 0.5
                stagnation_counter = 0
                
            # Convergence check: step size exhaustion
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
