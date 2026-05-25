# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES-inspired local search algorithm using rank-based population sampling.
# Search state: Maintains a current mean vector and a step-size (sigma) that scales with the search space.
# Candidate generation: Generates a population of points via multivariate normal sampling centered at the current mean.
# Selection and replacement: Selects the top 50% of the population to compute a new weighted mean.
# Adaptation: Updates the mean based on the best performers; sigma decays linearly to refine the search.
# Exploration mechanisms: Initial large sigma provides global coverage; stochastic sampling ensures diversity.
# Exploitation mechanisms: Mean shifting towards superior regions and shrinking sigma for local convergence.
# Boundary handling: Clamps candidates within the function's bounds using a clipping strategy.
# Budget strategy: Consumes exactly the budget by calculating population size proportional to remaining evaluations.
# Closest known influences: Simplified Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
# Novelty or unusual aspects: Minimalist implementation of rank-based evolution without full matrix decomposition.
# Failure modes: Slow convergence on highly non-convex or high-dimensional landscapes; vulnerable to premature convergence.
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
        mean = np.random.uniform(lb, ub, self.dim)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        
        evals_remaining = self.budget
        pop_size = min(evals_remaining, 4 + int(3 * np.log(self.dim)))
        
        while evals_remaining >= pop_size and pop_size > 0:
            # Generate population
            candidates = np.random.normal(mean, sigma, (pop_size, self.dim))
            candidates = np.clip(candidates, lb, ub)
            
            # Evaluate
            scores = np.array([func(x) for x in candidates])
            evals_remaining -= pop_size
            
            # Track global best
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_y:
                best_y = scores[min_idx]
                best_x = candidates[min_idx]
            
            # Selection: top 50%
            sorted_indices = np.argsort(scores)
            elite_indices = sorted_indices[:max(1, pop_size // 2)]
            
            # Adaptation: update mean and shrink step size
            mean = np.mean(candidates[elite_indices], axis=0)
            sigma *= 0.95
            
            # Adjust population size for final iterations
            pop_size = min(evals_remaining, pop_size)
            if pop_size < 2 and evals_remaining > 0:
                # Last remaining evaluations
                if evals_remaining > 0:
                    x = np.clip(mean + np.random.normal(0, sigma), lb, ub)
                    y = func(x)
                    evals_remaining -= 1
                    if y < best_y:
                        best_y = y
                        best_x = x
                break

        return best_x, best_y
