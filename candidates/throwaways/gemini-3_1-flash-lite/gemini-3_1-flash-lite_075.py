# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm with adaptive step size control (1+1-ES variant).
# Search state: Maintains the current best solution (center) and a step size (sigma).
# Candidate generation: Generates a single candidate point by adding Gaussian noise to the center.
# Selection and replacement: Simple greedy selection; if the new point is better, move the center.
# Adaptation: Uses the 1/5 success rule to adjust sigma: increase if success rate is high, decrease if low.
# Exploration mechanisms: Gaussian mutation controlled by sigma.
# Exploitation mechanisms: Local movement towards lower objective values.
# Boundary handling: Clamping candidate values to provided bounds.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: (1+1)-ES (Evolution Strategy).
# Novelty or unusual aspects: Extremely compact, dependency-free implementation for black-box optimization.
# Failure modes: Can get trapped in narrow local minima; performance degrades in very high dimensions.
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
        
        # Initialize
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        budget_left = self.budget - 1
        
        # Hyperparameters for (1+1)-ES
        sigma = 0.2 * (ub - lb)
        successes = 0
        
        best_x = np.copy(x)
        best_y = y
        
        while budget_left > 0:
            # Generate candidate
            z = np.random.normal(0, 1, self.dim)
            candidate = np.clip(x + sigma * z, lb, ub)
            
            # Evaluate
            new_y = func(candidate)
            budget_left -= 1
            
            # Selection
            if new_y <= y:
                x = candidate
                y = new_y
                successes += 1
                if y < best_y:
                    best_y = y
                    best_x = np.copy(x)
            
            # Adaptation (1/5 success rule) every min(dim, 100) evaluations
            evals_per_gen = max(1, self.dim)
            if (self.budget - budget_left) % evals_per_gen == 0:
                success_ratio = successes / evals_per_gen
                if success_ratio > 0.2:
                    sigma *= 1.2
                elif success_ratio < 0.2:
                    sigma *= 0.8
                successes = 0
                
            if budget_left <= 0:
                break
                
        return best_x, best_y
