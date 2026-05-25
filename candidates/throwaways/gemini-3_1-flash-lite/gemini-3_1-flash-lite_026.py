# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A CMA-ES inspired (1+1)-ES with self-adaptive step size control.
# Search state: Current best point (x) and scalar step size (sigma).
# Candidate generation: Gaussian perturbation of the current state: x_new = x + sigma * N(0, I).
# Selection and replacement: Deterministic selection; update current state if f(x_new) <= f(x).
# Adaptation: One-fifth success rule; increase sigma if success rate is high, decrease if low.
# Exploration mechanisms: Isotropic Gaussian mutation centered on the best found point.
# Exploitation mechanisms: Direct local search via successful mutations and step-size decay.
# Boundary handling: Clamping to provided bounds post-mutation.
# Budget strategy: Iterative loop until evaluation count exactly reaches budget.
# Closest known influences: (1+1)-Evolution Strategy with 1/5th success rule.
# Novelty or unusual aspects: Minimalist implementation of adaptive local search.
# Failure modes: Can get stuck in local optima; performance sensitive to initial sigma and budget size.
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

        # Initial state: center of the domain
        curr_x = (lb + ub) / 2.0
        curr_y = func(curr_x)
        
        # Adaptive step size initialization
        sigma = (ub - lb) / 4.0
        
        best_x = curr_x.copy()
        best_y = curr_y
        
        evals = 1
        success_count = 0
        
        while evals < self.budget:
            # Generate candidate mutation
            candidate = curr_x + np.random.normal(0, sigma, self.dim)
            candidate = np.clip(candidate, lb, ub)
            
            cand_y = func(candidate)
            evals += 1
            
            # Selection
            if cand_y <= curr_y:
                curr_x = candidate
                curr_y = cand_y
                success_count += 1
                if cand_y < best_y:
                    best_y = cand_y
                    best_x = candidate.copy()
            
            # 1/5th Success Rule Adaptation
            # Adjust sigma every 10 iterations (or dim*2)
            if evals % (self.dim * 2) == 0:
                success_rate = success_count / (self.dim * 2)
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                success_count = 0
                
            # Convergence check: stop if sigma is infinitesimal
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
