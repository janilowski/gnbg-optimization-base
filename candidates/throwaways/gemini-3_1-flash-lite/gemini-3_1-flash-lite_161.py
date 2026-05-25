# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free covariance matrix adaptation evolution strategy (CMA-ES) variant utilizing a simplified CMA approach.
# Search state: Maintains a mean vector, a global step size (sigma), and an identity-based covariance matrix (isotropic search).
# Candidate generation: Generates a population of candidate points sampled from a multivariate normal distribution centered at the mean.
# Selection and replacement: Selects the top-performing fraction of the population to update the mean.
# Adaptation: Updates the mean using a weighted average of successful candidates and adjusts step size via a simple 1/5th success rule.
# Exploration mechanisms: Samples from a normal distribution scaled by sigma; noise is injected via the initial spread and sigma.
# Exploitation mechanisms: The mean moves toward the center of the best-performing candidates, shrinking the search radius as convergence occurs.
# Boundary handling: Projects candidate vectors back into the feasible domain using clipping (clamping).
# Budget strategy: Divides budget into generations of fixed size; terminates once the budget is exhausted.
# Closest known influences: Simplified CMA-ES and (mu, lambda)-ES.
# Novelty or unusual aspects: Employs a robust, memory-efficient approach suitable for black-box benchmarks without complex covariance matrix updates.
# Failure modes: Can get trapped in local optima for highly multi-modal functions or fail if the initial sigma is poorly scaled.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.max_gen_size = min(20 + int(3 * np.log(dim)), budget // 10 if budget > 100 else 5)
        self.mu = self.max_gen_size // 2

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize search state
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        evals = 0
        
        while evals < self.budget:
            # Generate population
            pop_size = min(self.max_gen_size, self.budget - evals)
            candidates = np.random.normal(0, 1, (pop_size, self.dim)) * sigma + mean
            candidates = np.clip(candidates, lb, ub)
            
            # Evaluate
            scores = np.array([func(x) for x in candidates])
            evals += pop_size
            
            # Track global best
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_y:
                best_y = scores[min_idx]
                best_x = candidates[min_idx]
            
            # Selection: top mu individuals
            indices = np.argsort(scores)[:self.mu]
            parents = candidates[indices]
            
            # Adaptation: move mean and adjust sigma
            new_mean = np.mean(parents, axis=0)
            
            # Simple 1/5th success rule for sigma adaptation
            if np.mean(scores[:self.mu]) < best_y:
                sigma *= 1.1
            else:
                sigma *= 0.9
            
            mean = new_mean
            
            # Terminate if convergence is too tight
            if np.all(sigma < 1e-10):
                break
                
        return best_x, best_y
