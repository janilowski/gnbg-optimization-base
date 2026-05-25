# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free adaptive random search using a localized Gaussian mutation strategy.
# Search state: Maintains a single best-found candidate (elite) and a dynamic step-size (sigma).
# Candidate generation: Generates new candidates by sampling from a multivariate Gaussian centered at the current best, scaled by sigma.
# Selection and replacement: Simple elitist replacement; if a new candidate performs better, it replaces the current best.
# Adaptation: A 1/5th success rule heuristic: if the success rate is > 20%, sigma increases; if < 20%, sigma decreases.
# Exploration mechanisms: Initial sigma is large (fraction of domain range) to allow global search, shrinking over time.
# Exploitation mechanisms: Localized Gaussian sampling around the best known point for refinement.
# Boundary handling: candidates are clipped to [lb, ub] before evaluation.
# Budget strategy: Uniformly consumes the provided budget; if budget is high, sigma adapts to fine-tune.
# Closest known influences: (1+1)-ES (Evolution Strategy) with a self-adapting step size.
# Novelty or unusual aspects: Minimalist implementation of the 1/5th success rule for robust convergence in black-box scenarios.
# Failure modes: Can get trapped in local optima in highly multi-modal landscapes with very high dimensionality.
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
        
        # Initialize
        curr_x = np.random.uniform(lb, ub, self.dim)
        curr_y = func(curr_x)
        self.budget -= 1
        
        best_x, best_y = np.copy(curr_x), curr_y
        
        # Initial step size: 10% of domain range
        sigma = 0.1 * (ub - lb)
        success_count = 0
        
        # Iteration tracking for 1/5th rule
        k = 5
        
        while self.budget > 0:
            # Generate mutation
            candidate = curr_x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            y = func(candidate)
            self.budget -= 1
            
            # Selection
            if y < curr_y:
                curr_x, curr_y = candidate, y
                success_count += 1
                if y < best_y:
                    best_x, best_y = np.copy(candidate), y
            
            # Adaptation of sigma every k steps
            if (self.budget + 1) % k == 0:
                if success_count / k > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                success_count = 0
                
        return best_x, best_y
