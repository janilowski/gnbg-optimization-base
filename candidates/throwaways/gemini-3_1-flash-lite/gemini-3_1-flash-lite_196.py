# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (1+1)-Evolution Strategy with adaptive step-size control (1/5th success rule).
# Search state: Maintains a single candidate point (best_x) and a global step-size (sigma).
# Candidate generation: Generates a new point by adding Gaussian noise to the current best_x, scaled by sigma.
# Selection and replacement: Deterministic replacement; the new candidate replaces the incumbent if it yields a lower function value.
# Adaptation: The step-size sigma is updated every N=10 iterations: increased if success rate > 1/5, decreased otherwise.
# Exploration mechanisms: Driven by isotropic Gaussian mutation (sigma).
# Exploitation mechanisms: Local search around the current best estimate.
# Boundary handling: Points are projected back into the search space using clipping.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: (1+1)-ES, Rechenberg's 1/5th success rule.
# Novelty or unusual aspects: None; chosen for robustness and simplicity as a black-box optimizer.
# Failure modes: Can get trapped in local optima; ineffective for highly multimodal landscapes with sparse basins.
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
        x = np.random.uniform(lb, ub)
        y = func(x)
        self.budget -= 1
        
        sigma = 0.2 * (ub - lb)
        best_x, best_y = x.copy(), y
        
        successes = 0
        n_iters = 0
        
        while self.budget > 0:
            # Generate mutation
            z = best_x + np.random.normal(0, sigma)
            # Boundary constraint: clip to bounds
            z = np.clip(z, lb, ub)
            
            zy = func(z)
            self.budget -= 1
            n_iters += 1
            
            # Selection
            if zy <= best_y:
                best_x, best_y = z, zy
                successes += 1
            
            # Adaptive step-size control (1/5th success rule) every 10 iterations
            if n_iters >= 10:
                if successes / n_iters > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                n_iters = 0
                successes = 0
                
            if self.budget <= 0:
                break
                
        return best_x, best_y
