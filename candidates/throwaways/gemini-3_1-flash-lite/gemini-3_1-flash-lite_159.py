# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk.
# Search state: Tracks the current best solution and a dynamic step size (sigma).
# Candidate generation: Generates candidates via Gaussian mutation of the current best.
# Selection and replacement: Simple greedy update: replaces the best if the candidate is better.
# Adaptation: Employs a 1/5 success rule: increases sigma if success rate > 0.2, decreases if < 0.2.
# Exploration mechanisms: Large initial sigma and random direction sampling.
# Exploitation mechanisms: Shrinking sigma allows refinement of the local optimum.
# Boundary handling: Clamping candidates to the objective function's bounds.
# Budget strategy: Exhausts the entire budget linearly until convergence or termination.
# Closest known influences: Evolutionary Strategy (1+1)-ES.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness and budget awareness.
# Failure modes: Can get trapped in local optima for highly multi-modal functions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        else:
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Initialize
        x = np.random.uniform(lb, ub)
        y = func(x)
        budget_left = self.budget - 1
        
        best_x = np.copy(x)
        best_y = y
        
        # Initial sigma (step size) as 20% of range
        sigma = 0.2 * (ub - lb)
        
        # Adaptation parameters for 1/5 success rule
        success_count = 0
        gen_count = 0
        
        while budget_left > 0:
            # Generate candidate via Gaussian mutation
            candidate = x + np.random.normal(0, sigma)
            # Boundary clamping
            candidate = np.clip(candidate, lb, ub)
            
            val = func(candidate)
            budget_left -= 1
            gen_count += 1
            
            # Greedy replacement
            if val < y:
                x = np.copy(candidate)
                y = val
                success_count += 1
                if y < best_y:
                    best_x = np.copy(x)
                    best_y = y
            
            # Update sigma every 10 generations (1/5 success rule)
            if gen_count >= 10:
                if success_count / gen_count > 0.2:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                success_count = 0
                gen_count = 0
                
        return best_x, best_y
