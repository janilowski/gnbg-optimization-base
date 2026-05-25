# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (A)daptive (S)tep-size (S)caling.
# Search state: Tracks the current best point found so far and an adaptive step-size (sigma).
# Candidate generation: Generates candidates via isotropic Gaussian mutation around the current best.
# Selection and replacement: Simple elitist replacement; if a candidate improves the best, update best.
# Adaptation: Employs a 1/5th success rule: if the success rate is high, increase step-size; if low, decrease it.
# Exploration mechanisms: Initially high sigma allows wide search; sigma decays/adapts as the search focuses.
# Exploitation mechanisms: Local search around the current optimum using the success-rate adapted step-size.
# Boundary handling: Clamping candidates to the provided function bounds.
# Budget strategy: Iterates until the evaluation count matches the budget; stops immediately if reached.
# Closest known influences: Evolutionary Strategies, specifically the 1/5th success rule of Rechenberg.
# Novelty or unusual aspects: Minimalist implementation without complex matrix covariance updates.
# Failure modes: Can get stuck in local optima if the search space is highly multi-modal and the initial sigma is too small.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x_best = np.random.uniform(lb, ub, self.dim)
        y_best = func(x_best)
        evals_used = 1
        
        # Adaptive step size parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        while evals_used < self.budget:
            # Generate candidate via Gaussian mutation
            x_cand = x_best + np.random.normal(0, sigma, self.dim)
            x_cand = np.clip(x_cand, lb, ub)
            
            y_cand = func(x_cand)
            evals_used += 1
            
            # Elitist selection
            if y_cand < y_best:
                x_best = x_cand
                y_best = y_cand
                success_count += 1
            
            # Step size adaptation (1/5th success rule logic)
            if evals_used % 10 == 0:
                success_rate = success_count / 10.0
                if success_rate > 0.2:
                    sigma *= 1.1  # Expand search
                elif success_rate < 0.2:
                    sigma *= 0.9  # Focus search
                success_count = 0
                
        return x_best, y_best
