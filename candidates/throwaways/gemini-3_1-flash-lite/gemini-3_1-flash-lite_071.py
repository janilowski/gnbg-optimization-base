# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (1+1)-Evolution Strategy with adaptive step-size control.
# Search state: Maintains a single candidate 'x' and its corresponding fitness 'y', plus a global step-size 'sigma'.
# Candidate generation: Generates new samples by adding Gaussian noise scaled by 'sigma' to the best observed point.
# Selection and replacement: Simple greedy update; if the new point is better, it replaces the current best.
# Adaptation: Uses the 1/5th success rule to increase or decrease 'sigma' based on recent success frequency.
# Exploration mechanisms: Stochastic Gaussian perturbation; step-size control ensures search radius oscillates.
# Exploitation mechanisms: Local improvement via hill-climbing behavior near the current best.
# Boundary handling: Points exceeding bounds are clipped to the nearest valid value within the defined feasible domain.
# Budget strategy: Iterates exactly within the provided budget; stops when 'budget' is reached.
# Closest known influences: Classical (1+1)-ES with 1/5th success rule.
# Novelty or unusual aspects: Extremely compact implementation utilizing standard library and numpy.
# Failure modes: Susceptible to local optima; struggles with high-dimensional landscapes where the optimal step size varies significantly by region.
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
        
        # Initialize
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        
        best_x = np.copy(x)
        best_y = y
        
        evals = 1
        sigma = 0.2 * (ub - lb)
        success_history = []
        
        while evals < self.budget:
            # Generate candidate
            candidate = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
            c_y = func(candidate)
            evals += 1
            
            # Selection/Replacement
            if c_y < best_y:
                best_y = c_y
                best_x = np.copy(candidate)
                success_history.append(1)
            else:
                success_history.append(0)
            
            # 1/5th success rule adaptation
            if len(success_history) >= 20:
                success_ratio = sum(success_history) / len(success_history)
                if success_ratio > 0.2:
                    sigma *= 1.1
                elif success_ratio < 0.2:
                    sigma /= 1.1
                success_history = []
                
        return best_x, best_y
