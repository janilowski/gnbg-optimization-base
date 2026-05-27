# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) derivative using a simplified 1+1 evolution strategy approach.
# Search state: Tracks the current best point, the step-size (sigma), and the search distribution mean.
# Candidate generation: Generates candidates via multivariate normal sampling centered at the best point, scaled by sigma.
# Selection and replacement: Deterministic replacement (plus-strategy): only the new solution is kept if it performs better than the current best.
# Adaptation: Uses 1/5th success rule to adapt the step-size: increases sigma upon successful moves and decreases upon failures.
# Exploration mechanisms: Gaussian mutation (exploration) is modulated by the adaptive step-size sigma.
# Exploitation mechanisms: Local hill-climbing around the current best candidate.
# Boundary handling: Clamps candidates back into the search space via projection (clipping).
# Budget strategy: Exhausts the total budget by continuously performing local mutations.
# Closest known influences: (1+1)-ES with the 1/5th success rule.
# Novelty or unusual aspects: Extremely compact implementation utilizing NumPy for vectorization.
# Failure modes: Can get stuck in local optima; performance depends heavily on the initial sigma and the objective landscape.
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
        sigma = 0.2 * (ub - lb)
        x = np.random.uniform(lb, ub)
        y = func(x)
        
        best_x, best_y = x.copy(), y
        evals = 1
        
        # Counters for the 1/5th success rule
        success_count = 0
        gen_count = 0
        
        while evals < self.budget:
            # Generate candidate
            z = np.random.normal(0, 1, self.dim)
            x_new = np.clip(best_x + sigma * z, lb, ub)
            
            y_new = func(x_new)
            evals += 1
            gen_count += 1
            
            # Selection
            if y_new < best_y:
                best_x, best_y = x_new, y_new
                success_count += 1
            
            # Adapt sigma periodically (every 5 iterations)
            if gen_count >= 5:
                ratio = success_count / gen_count
                if ratio > 0.2:
                    sigma *= 1.25
                elif ratio < 0.2:
                    sigma *= 0.8
                
                # Reset counters
                success_count = 0
                gen_count = 0
                
            # Exit if remaining budget is insufficient for meaningful change
            if evals >= self.budget:
                break
                
        return best_x, best_y
