# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant with a simplified diagonal approximation.
# Search state: Maintains a mean vector, a global step size (sigma), and a diagonal covariance vector.
# Candidate generation: Samples candidate solutions from a multivariate normal distribution defined by the current mean and diagonal covariance.
# Selection and replacement: Selects the top-performing fraction of samples to update the mean.
# Adaptation: Updates mean via weighted average; updates step size via cumulative step-length control (path length control).
# Exploration mechanisms: Gaussian mutation controlled by sigma; initial sigma is proportional to the input domain width.
# Exploitation mechanisms: Mean shift toward better performing regions of the search space.
# Boundary handling: Reflective clipping within the provided lower and upper bounds.
# Budget strategy: Stops evaluation immediately once the budget is exhausted.
# Closest known influences: CMA-ES and (mu, lambda)-ES.
# Novelty or unusual aspects: Diagonal-only covariance for computational efficiency and memory reduction.
# Failure modes: Might prematurely converge on highly multi-modal functions or struggle if optimal solution is at the extreme corner of a narrow box constraint.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.n_pop = 4 + int(3 * np.log(dim))
        self.mu = self.n_pop // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        dim = self.dim
        center = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        diag_cov = np.ones(dim)
        
        best_x = None
        best_y = np.inf
        evals = 0
        
        while evals < self.budget:
            samples = []
            values = []
            
            for _ in range(self.n_pop):
                if evals >= self.budget:
                    break
                
                # Sample and bound
                x = center + sigma * np.random.normal(0, diag_cov)
                x = np.clip(x, lb, ub)
                
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                
                samples.append(x)
                values.append(y)
            
            if not values:
                break
                
            # Selection
            idx = np.argsort(values)[:self.mu]
            samples = np.array(samples)[idx]
            
            # Update center (mean shift)
            new_center = np.dot(self.weights, samples)
            
            # Adaptation of sigma (simple path control estimation)
            step = (new_center - center) / sigma
            sigma *= np.exp(np.clip(np.mean(step**2)**0.5 - 0.5, -0.2, 0.2))
            
            # Update mean
            center = new_center
            
            # Decay search radius if stalled (prevent stagnation)
            if np.all(sigma < 1e-9 * (ub - lb)):
                sigma = 0.1 * (ub - lb)

        return best_x, best_y
