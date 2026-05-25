# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (X, sigma)-CMA-ES variant that uses a rank-based evolution strategy.
# Search state: Maintains a current mean vector, a step-size (sigma), and a diagonal covariance matrix.
# Candidate generation: Samples multivariate normal distributions centered at the mean.
# Selection and replacement: Selects the best percentile of candidates to update the distribution parameters.
# Adaptation: Updates mean via weighted average of best samples; updates sigma via simple path length control.
# Exploration mechanisms: Stochastic sampling regulated by the global sigma parameter.
# Exploitation mechanisms: Mean shifting towards successful regions and rank-based selection pressure.
# Boundary handling: Clipping candidates to the feasible domain defined by the objective function.
# Budget strategy: Stops sampling once the total function evaluation count reaches the provided budget.
# Closest known influences: Simplified CMA-ES / Estimation of Distribution Algorithm (EDA).
# Novelty or unusual aspects: Minimalist implementation focusing on robustness in low-to-medium dimensions.
# Failure modes: Slow convergence on highly non-convex or needle-in-haystack landscapes; step-size may collapse prematurely.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        
    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
            
        # Initialization
        x = np.random.uniform(lb, ub, self.dim)
        sigma = 0.3 * (ub - lb)
        best_x = np.copy(x)
        best_y = func(x)
        evals = 1
        
        # Evolution loop
        while evals < self.budget:
            # Generate offspring
            candidates = []
            values = []
            
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and clip
                z = np.random.normal(0, 1, self.dim)
                curr_x = np.clip(x + sigma * z, lb, ub)
                
                y = func(curr_x)
                evals += 1
                
                candidates.append(curr_x)
                values.append(y)
                
                # Track global best
                if y < best_y:
                    best_y = y
                    best_x = np.copy(curr_x)
            
            if not values:
                break
                
            # Selection: Sort by performance
            idx = np.argsort(values)
            n_best = max(1, self.pop_size // 2)
            elite_x = np.array([candidates[i] for i in idx[:n_best]])
            
            # Adaptation: Move mean towards elite average
            old_x = np.copy(x)
            x = np.mean(elite_x, axis=0)
            
            # Simple step-size adaptation (increase if successful, decrease if stagnated)
            if np.mean(values[:n_best]) < best_y:
                sigma *= 1.1
            else:
                sigma *= 0.95
                
            # Keep sigma within bounds
            sigma = np.clip(sigma, 1e-5 * (ub - lb), 1.0 * (ub - lb))
            
        return best_x, best_y
