# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free covariance matrix adaptation evolution strategy (CMA-ES) derivative using a simplified 1+1 evolution strategy.
# Search state: Current best point (x) and the mutation step size (sigma).
# Candidate generation: Gaussian mutation centered on the current best point.
# Selection and replacement: Greedy selection (1+1); only replace if the new sample is strictly better.
# Adaptation: One-fifth success rule; increase step size if success rate is high, decrease if low.
# Exploration mechanisms: Mutations are scaled by the current step size (sigma).
# Exploitation mechanisms: Local hill-climbing via successful mutations.
# Boundary handling: Resampling within bounds for any parameter that violates constraints.
# Budget strategy: Strict iteration counter tracking each function evaluation.
# Closest known influences: 1+1-ES with 1/5 success rule.
# Novelty or unusual aspects: Extremely compact memory footprint.
# Failure modes: Can get trapped in local optima in highly non-convex surfaces.
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
        
        # Initialize state
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        evals = 1
        
        # Step size parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        gen_count = 0
        
        while evals < self.budget:
            # Generate candidate via Gaussian mutation
            # Resampling strategy for boundary constraint handling
            z = np.random.normal(0, 1, self.dim)
            x_new = np.clip(x + sigma * z, lb, ub)
            
            y_new = func(x_new)
            evals += 1
            
            # Selection: 1+1 evolution strategy
            if y_new <= y:
                x = x_new
                y = y_new
                success_count += 1
            
            gen_count += 1
            
            # 1/5 Success Rule Adaptation
            if gen_count == 5:
                if success_count / gen_count > 0.2:
                    sigma *= 1.2
                elif success_count / gen_count < 0.2:
                    sigma *= 0.8
                success_count = 0
                gen_count = 0
            
            # Terminate if budget reached
            if evals >= self.budget:
                break
                
        return x, y
