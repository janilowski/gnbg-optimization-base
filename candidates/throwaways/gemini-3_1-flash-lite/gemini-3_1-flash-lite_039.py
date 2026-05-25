# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A lightweight implementation of a (1+1)-Evolution Strategy with adaptive step-size control (1/5th rule).
# Search state: Maintains a single current best solution (x) and the standard deviation (sigma) of mutation.
# Candidate generation: Generates a single candidate point by adding Gaussian noise to the current best.
# Selection and replacement: The candidate replaces the current best if it achieves a lower function value.
# Adaptation: The mutation strength sigma is updated every 10 iterations: increased if success rate > 0.2, decreased otherwise.
# Exploration mechanisms: Gaussian mutation provides local neighborhood exploration.
# Exploitation mechanisms: The algorithm narrows the search space as sigma decays, refining the current best.
# Boundary handling: Candidates are clipped to the provided function bounds before evaluation.
# Budget strategy: Iteratively evaluates until the budget is exhausted.
# Closest known influences: Rechenberg’s (1+1)-ES with the 1/5th success rule.
# Novelty or unusual aspects: Minimalist loop structure optimized for black-box environments with unknown landscape gradients.
# Failure modes: Can get stuck in local optima; performance is sensitive to the initial sigma value.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x = np.random.uniform(lb, ub)
        y = func(x)
        self.budget -= 1
        
        best_x, best_y = np.copy(x), y
        
        # Adaptive parameters
        sigma = (ub - lb) / 5.0
        successes = 0
        iterations = 0
        
        # Main optimization loop
        while self.budget > 0:
            iterations += 1
            
            # Generate candidate via Gaussian mutation
            candidate = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
            
            # Evaluate
            candidate_y = func(candidate)
            self.budget -= 1
            
            # Selection
            if candidate_y < best_y:
                best_x, best_y = candidate, candidate_y
                successes += 1
            
            # Adaptive step-size control (1/5th rule) every 10 iterations
            if iterations % 10 == 0:
                if successes / 10 > 0.2:
                    sigma *= 1.2
                else:
                    sigma /= 1.2
                successes = 0
                
            if self.budget <= 0:
                break
                
        return best_x, best_y
