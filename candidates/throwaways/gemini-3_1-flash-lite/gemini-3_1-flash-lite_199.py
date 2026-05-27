# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (A)daptive (S)tep size (SAS).
# Search state: Maintains the best known position (best_x) and a global step size (sigma).
# Candidate generation: Generates candidates via isotropic Gaussian mutation around the best_x.
# Selection and replacement: Greedy selection; if a candidate improves the function, it replaces best_x.
# Adaptation: The "1/5th success rule" is used to adapt sigma: increase if success rate > 0.2, decrease otherwise.
# Exploration mechanisms: Initial sigma is large (fraction of domain range), controlled by the adaptive rule.
# Exploitation mechanisms: Local hill-climbing via Gaussian mutation and sigma decay.
# Boundary handling: Candidates are clipped to the function bounds.
# Budget strategy: Exhausts the provided evaluation budget exactly through a fixed loop.
# Closest known influences: Evolutionary Strategies, specifically $(1+1)$-ES.
# Novelty or unusual aspects: Extremely compact implementation with reactive step-size adjustment.
# Failure modes: Can get trapped in narrow local minima for highly multi-modal functions.
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
        
        range_width = ub - lb
        
        # Initialize state
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        
        # Hyperparameters for step size adaptation
        sigma = 0.2 * range_width
        success_count = 0
        evals = 1
        
        # Run until budget is exhausted
        while evals < self.budget:
            # Generate candidate
            candidate = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
            
            y = func(candidate)
            evals += 1
            
            # Selection
            if y < best_y:
                best_x, best_y = candidate, y
                success_count += 1
            
            # Periodically adapt step size (every 10 evals or when progress is slow)
            if evals % 10 == 0:
                # 1/5th success rule: aim for 20% success rate
                if success_count / 10 > 0.2:
                    sigma *= 1.1  # Expand
                else:
                    sigma *= 0.9  # Contract
                success_count = 0
                
            # Convergence check: stop if step size is effectively zero
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
