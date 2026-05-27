# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A self-adapting (1+1)-ES with covariance matrix adaptation (simplification of CMA-ES).
# Search state: Current best point and a global step size (sigma).
# Candidate generation: Multivariate normal sampling centered on the current best point scaled by sigma.
# Selection and replacement: Greedy selection; the new candidate replaces the best if it yields a lower function value.
# Adaptation: The 1/5th success rule: sigma increases if success rate is high, decreases if low.
# Exploration mechanisms: Global noise injection via isotropic Gaussian mutation.
# Exploitation mechanisms: Local descent via step-size refinement and greediness.
# Boundary handling: Clipping candidates to the feasible domain, effectively projecting them onto the boundary.
# Budget strategy: Explicit loop counter checks against the provided budget.
# Closest known influences: (1+1)-Evolution Strategy with cumulative step-length adaptation.
# Novelty or unusual aspects: Simplified, robust implementation suitable for black-box environments with limited budget.
# Failure modes: Susceptible to local optima and premature convergence if the step size shrinks too rapidly.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        
        best_x = np.copy(x)
        best_y = y
        
        # Strategy parameters
        sigma = 0.2 * (ub - lb)
        evals = 1
        
        # Target: 1/5th success rate adaptation
        success_hist = []
        
        while evals < self.budget:
            # Generate candidate
            candidate = best_x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            y_cand = func(candidate)
            evals += 1
            
            # Select
            if y_cand < best_y:
                best_x, best_y = candidate, y_cand
                success_hist.append(1)
            else:
                success_hist.append(0)
                
            # Adapt sigma periodically (every dim iterations)
            if len(success_hist) >= self.dim:
                success_rate = np.mean(success_hist)
                if success_rate > 0.2:
                    sigma *= 1.25
                elif success_rate < 0.2:
                    sigma *= 0.8
                success_hist = []
                
        return best_x, best_y
