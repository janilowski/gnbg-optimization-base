# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (1+1)-CMA-ES variant using isotropic self-adaptive step size.
# Search state: Stores current incumbent mean (x), step size (sigma), and success history.
# Candidate generation: Multivariate normal mutation centered at the current incumbent.
# Selection and replacement: Greedy selection; incumbent is updated if the offspring is better.
# Adaptation: The step size (sigma) increases if the last mutation was successful and decreases otherwise (1/5th success rule).
# Exploration mechanisms: Stochastic perturbation controlled by the step size sigma.
# Exploitation mechanisms: Local hill climbing via coordinate-wise refinement.
# Boundary handling: Clamping offspring to the feasible domain.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: (1+1)-ES with 1/5th success rule adaptation.
# Novelty or unusual aspects: Minimalist implementation suitable for small-to-medium black-box constraints.
# Failure modes: Susceptible to local optima; struggles with high-dimensional non-convex landscapes.
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

        # Initialization
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        evals = 1
        
        # Initial step size (1/10th of search range)
        sigma = 0.1 * (ub - lb)
        
        # Success rule parameters
        success_history = []
        
        while evals < self.budget:
            # Generate offspring
            z = np.random.normal(0, 1, self.dim)
            x_new = np.clip(x + sigma * z, lb, ub)
            
            y_new = func(x_new)
            evals += 1
            
            # Selection
            if y_new <= y:
                x = x_new
                y = y_new
                success_history.append(1)
            else:
                success_history.append(0)
            
            # Adapt sigma periodically (1/5th success rule)
            # Update every 'dim' steps for stability
            if len(success_history) >= self.dim:
                success_rate = sum(success_history) / len(success_history)
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                success_history = []
            
            # Termination check for small sigma
            if np.all(sigma < 1e-12):
                break
                
        return x, y
