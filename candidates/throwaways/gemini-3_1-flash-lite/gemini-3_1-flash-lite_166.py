# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Step-Size (ASSA).
# Search state: Current best point and a scalar step-size.
# Candidate generation: Gaussian perturbation of the current best point scaled by the step-size.
# Selection and replacement: Greedy selection (only improves).
# Adaptation: Success-rule (1/5th rule): halve step-size on failure, multiply by 1.2 on success.
# Exploration mechanisms: Large initial step-size covering the entire search space.
# Exploitation mechanisms: Contractive step-size adaptation upon successful local improvements.
# Boundary handling: Clipping candidates to the defined box constraints.
# Budget strategy: Iterative loop until the evaluation count reaches the budget.
# Closest known influences: Adaptive Random Search, 1/5th success rule of Evolution Strategies.
# Novelty or unusual aspects: Extremely lightweight, memory-efficient, and robust convergence for uni-modal landscapes.
# Failure modes: Can get stuck in local optima; ineffective for highly non-convex/discontinuous landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract boundaries
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize
        x_best = np.random.uniform(lb, ub, self.dim)
        y_best = func(x_best)
        evals = 1
        
        # Initial step size (10% of range)
        step_size = 0.1 * (ub - lb)
        
        while evals < self.budget:
            # Generate candidate
            x_cand = x_best + np.random.normal(0, step_size, self.dim)
            x_cand = np.clip(x_cand, lb, ub)
            
            y_cand = func(x_cand)
            evals += 1
            
            # Selection/Adaptation (1/5th rule idea)
            if y_cand < y_best:
                x_best, y_best = x_cand, y_cand
                step_size *= 1.2  # Expand on success
            else:
                step_size *= 0.5  # Contract on failure
                
            # Stop if step size becomes negligible
            if np.all(step_size < 1e-12):
                break
                
        return x_best, y_best
