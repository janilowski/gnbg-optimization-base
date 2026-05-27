# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant with a simplified diagonal covariance structure.
# Search state: Maintains a distribution mean (center of mass) and a diagonal vector representing the step-size/spread in each dimension.
# Candidate generation: Generates a population of points by sampling from a multivariate normal distribution centered at the mean, constrained by the diagonal scale.
# Selection and replacement: Uses a rank-based selection where the best fraction of the population updates the mean via weighted averaging.
# Adaptation: Updates the mean based on successful candidates and dynamically adjusts the global step-size (sigma) based on the progress of the population.
# Exploration mechanisms: The initial high sigma and sampling distribution ensure broad coverage, contracting as convergence progresses.
# Exploitation mechanisms: The weighted mean update moves the search center towards high-quality regions of the search space.
# Boundary handling: Reflective clipping: points exceeding bounds are projected back to the boundary using a clamping method.
# Budget strategy: Iterative loop that terminates once the call count matches the budget; population size is scaled dynamically with dimension.
# Closest known influences: Simplified (diagonal) CMA-ES.
# Novelty or unusual aspects: Minimalist implementation of adaptive distribution matching without full covariance matrix inversion.
# Failure modes: Can get stuck in narrow local minima in highly non-convex, deceptive landscapes or if dimensions are highly correlated.
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
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        
        evals_left = self.budget
        
        while evals_left > 0:
            # Generate population
            pop_size = min(evals_left, self.pop_size)
            samples = np.random.normal(0, 1, (pop_size, self.dim)) * sigma + mean
            
            # Boundary handling: Clamping
            samples = np.clip(samples, lb, ub)
            
            # Evaluate
            scores = np.array([func(s) for s in samples])
            evals_left -= pop_size
            
            # Track global best
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_y:
                best_y = scores[min_idx]
                best_x = samples[min_idx].copy()
                
            # Selection: Sort by fitness
            order = np.argsort(scores)
            elite_count = max(1, pop_size // 2)
            elites = samples[order[:elite_count]]
            
            # Adaptation: Move mean toward best samples
            new_mean = np.mean(elites, axis=0)
            
            # Update step size based on success if mean moved significantly
            diff = new_mean - mean
            sigma = 0.95 * sigma + (0.05 * np.abs(diff))
            mean = new_mean
            
            # Heuristic stall prevention
            if np.all(sigma < 1e-9 * (ub - lb)):
                sigma = 0.2 * (ub - lb)
                
        return best_x, best_y
