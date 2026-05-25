# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free adaptive random search algorithm using Gaussian perturbations.
# Search state: Maintains the current best point and a dynamically adjusted step-size (sigma).
# Candidate generation: Generates candidates via isotropic Gaussian sampling centered at the current best point.
# Selection and replacement: Greedy replacement; the best point is updated if a candidate yields a lower objective value.
# Adaptation: Employs a 1/5th success rule to adjust sigma: increases if success rate is high, decreases if low.
# Exploration mechanisms: Initial sigma is set to a fraction of the search space, allowing broad coverage.
# Exploitation mechanisms: Sigma decreases over time to allow fine-tuning around the current local optimum.
# Boundary handling: Candidates are clipped to the provided search bounds using numpy.clip.
# Budget strategy: Iterates exactly until the budget is exhausted, with a warm-up phase for sigma adjustment.
# Closest known influences: Evolutionary strategies (1+1)-ES.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness and fast convergence on convex-like landscapes.
# Failure modes: Can get stuck in local optima on highly multimodal, deceptive landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = func.lower, func.upper
        else:
            lb, ub = func.bounds.lb, func.bounds.ub
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Initialize state
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        
        # Step size control parameters
        sigma = (ub - lb) / 5.0
        success_count = 0
        evals = 1
        
        # Iterative search
        while evals < self.budget:
            # Generate candidate
            candidate = np.random.normal(best_x, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            y = func(candidate)
            evals += 1
            
            # Selection
            if y < best_y:
                best_y = y
                best_x = candidate
                success_count += 1
            
            # Adaptation (1/5th success rule)
            if evals % 10 == 0:
                success_rate = success_count / 10.0
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                success_count = 0
                
            # Convergence check: shrink sigma if too large
            if np.all(sigma < (ub - lb) * 1e-9):
                sigma = (ub - lb) * 0.1
                
        return best_x, best_y
