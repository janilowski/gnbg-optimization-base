# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (A)daptive (S)tep size (SAS).
# Search state: Maintains the current best point found so far.
# Candidate generation: Gaussian perturbation centered on the current best point.
# Selection and replacement: Greedy selection; if a candidate improves the best point, replace it.
# Adaptation: Adaptive step-size (sigma) control using the 1/5th success rule.
# Exploration mechanisms: Global search via initial wide variance and localized exploration via adaptive Gaussian sampling.
# Exploitation mechanisms: Local hill climbing through iterative refinement of existing best solutions.
# Boundary handling: Hard clipping of generated candidates to the search space domain.
# Budget strategy: Uniform allocation across the total budget until exhaustion.
# Closest known influences: (1+1) Evolution Strategy.
# Novelty or unusual aspects: Minimalist implementation of adaptive step-size targeting black-box reliability.
# Failure modes: Can get stuck in local optima on highly multimodal landscapes early if the step size shrinks too rapidly.
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
        x_best = np.random.uniform(lb, ub, self.dim)
        y_best = func(x_best)
        budget_left = self.budget - 1
        
        # Adaptive step size (sigma)
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        # Iterate until budget depletion
        while budget_left > 0:
            # Generate candidate using current location + Gaussian noise
            noise = np.random.normal(0, sigma, self.dim)
            x_cand = np.clip(x_best + noise, lb, ub)
            
            y_cand = func(x_cand)
            budget_left -= 1
            
            # Selection
            if y_cand < y_best:
                x_best = x_cand
                y_best = y_cand
                success_count += 1
            
            # Adaptive step control (1/5th success rule)
            # Every 10 iterations, adjust sigma to maintain roughly 20% success rate
            if (self.budget - budget_left) % 10 == 0:
                if success_count / 10 > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                success_count = 0
                
        return x_best, y_best
