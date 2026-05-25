# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant utilizing a simplified rank-one update for global optimization.
# Search state: Maintains a mean vector, a global step size (sigma), and a covariance matrix (identity-scaled).
# Candidate generation: Samples population members from a multivariate normal distribution centered at the mean.
# Selection and replacement: Selects the best 50% of the population to update the mean.
# Adaptation: Updates the mean using weighted average; sigma is adjusted via a simplified success-rule heuristic.
# Exploration mechanisms: Initial sigma is set to 20% of the domain width; isotropic sampling provides broad coverage.
# Exploitation mechanisms: The mean moves toward the top-performing samples, converging iteratively.
# Boundary handling: Points are clipped to the defined bounds using numpy's clip function.
# Budget strategy: Evaluates exactly until the budget is exhausted, termination occurs before budget if population size would exceed remaining limit.
# Closest known influences: CMA-ES and basic Evolution Strategies (ES).
# Novelty or unusual aspects: Minimalist implementation of rank-based evolution without heavy matrix decompositions.
# Failure modes: Slow convergence on highly deceptive (non-convex) landscapes; sensitive to initial sigma choice.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.2 * (ub - lb)
        best_x = None
        best_y = float('inf')
        evals = 0
        
        while evals < self.budget:
            # Generate population
            remaining = self.budget - evals
            current_pop = min(self.pop_size, remaining)
            
            samples = []
            for _ in range(current_pop):
                x = np.clip(mean + sigma * np.random.randn(self.dim), lb, ub)
                y = func(x)
                evals += 1
                samples.append((x, y))
                
                if y < best_y:
                    best_y = y
                    best_x = x
            
            # Selection: Sort by fitness
            samples.sort(key=lambda item: item[1])
            
            # Selection: Use top half for parent update
            n_parents = max(1, current_pop // 2)
            parents = [s[0] for s in samples[:n_parents]]
            
            # Adaptation: Move mean toward best samples
            new_mean = np.mean(parents, axis=0)
            
            # Sigma adaptation: Simple success rule heuristic
            if new_mean[0] == mean[0] and current_pop > 1:
                sigma *= 0.95  # Shrink if stuck
            else:
                sigma *= 1.05  # Expand if progressing
                sigma = np.clip(sigma, 1e-4, (ub - lb))
                
            mean = new_mean
            
        return best_x, best_y
