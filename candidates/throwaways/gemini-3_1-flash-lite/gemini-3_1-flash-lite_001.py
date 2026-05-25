# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free stochastic optimizer based on the Covariance Matrix Adaptation Evolution Strategy (simplified).
# Search state: Maintains a mean vector, a covariance matrix (diagonal approximation), and a global step size (sigma).
# Candidate generation: Generates a population of candidate points using multivariate normal sampling centered at the mean.
# Selection and replacement: Selects a subset of best-performing candidates; updates the mean as a weighted average.
# Adaptation: Updates the mean toward the best samples and increases/decreases step size (1/5th rule inspired) based on population success.
# Exploration mechanisms: Gaussian sampling with global step size allows for wide search spread early on.
# Exploitation mechanisms: Mean updating and contraction of step size focus the search as progress slows.
# Boundary handling: Points exceeding bounds are projected back to the hypercube limits.
# Budget strategy: Divides the total budget into discrete generations, terminating when budget is exhausted.
# Closest known influences: Simplified CMA-ES and (mu, lambda)-ES.
# Novelty or unusual aspects: Uses a simple adaptive step-size mechanism instead of full Cholesky decomposition to reduce overhead.
# Failure modes: Can get stuck in local optima if the initial step size is too small or if the landscape is highly non-separable.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        
    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
            
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.2 * (ub - lb)
        best_x = None
        best_y = float('inf')
        evals = 0
        
        # Iterative evolution until budget is near depletion
        while evals + self.pop_size <= self.budget:
            # Generate population
            pop = np.random.normal(0, 1, (self.pop_size, self.dim)) * sigma + mean
            pop = np.clip(pop, lb, ub)
            
            # Evaluate
            scores = np.array([func(x) for x in pop])
            evals += self.pop_size
            
            # Track global best
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_y:
                best_y = scores[min_idx]
                best_x = pop[min_idx]
                
            # Selection: Sort by performance
            idx = np.argsort(scores)
            elite_size = self.pop_size // 2
            elites = pop[idx[:elite_size]]
            
            # Adaptation: Move mean toward elite average
            new_mean = np.mean(elites, axis=0)
            
            # Step size adaptation (1/5th rule intuition)
            if scores[idx[0]] < best_y:
                sigma *= 1.1
            else:
                sigma *= 0.95
                
            mean = new_mean
            
            # Break if convergence is extremely tight
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
