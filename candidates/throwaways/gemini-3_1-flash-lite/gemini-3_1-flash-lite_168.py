# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free covariance matrix adaptation evolution strategy (CMA-ES) variant limited to simple adaptive random search for compactness.
# Search state: Tracks the current best solution and a diagonal step-size (standard deviation) vector.
# Candidate generation: Samples new points from a multivariate normal distribution centered on the best known solution.
# Selection and replacement: Simple greedy replacement; if a sample improves the best solution, the best solution is updated.
# Adaptation: Updates the mutation step-size using a 1/5th success rule, scaling by expansion/contraction factors based on success rate.
# Exploration mechanisms: Gaussian mutation with a dynamic step size that expands during stagnation.
# Exploitation mechanisms: Local search around the current best candidate using adaptive step-size shrinking.
# Boundary handling: Clamping samples to the defined box constraints.
# Budget strategy: Uniformly partitions the budget by calculating iteration counts based on population size.
# Closest known influences: (1+1)-ES (Evolution Strategy).
# Novelty or unusual aspects: Minimal implementation without matrix inversion or complex covariance updates.
# Failure modes: Can get stuck in local optima if the initial step size is too small; ineffective for extremely high-dimensional non-separable landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.max_iter = budget // self.pop_size

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = func.lower, func.upper
        else:
            lb, ub = func.bounds.lb, func.bounds.ub
        
        lb = np.array(lb)
        ub = np.array(ub)
        span = ub - lb

        # Initialization
        best_x = lb + np.random.rand(self.dim) * span
        best_y = func(best_x)
        
        # Step size (sigma) initialized to 10% of the domain span
        sigma = 0.1 * span
        evals = 1
        
        for _ in range(self.max_iter):
            if evals >= self.budget:
                break
            
            # Generate offspring
            samples = best_x + np.random.normal(0, sigma, (self.pop_size, self.dim))
            # Clamp to bounds
            samples = np.clip(samples, lb, ub)
            
            success_count = 0
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                y = func(samples[i])
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = samples[i]
                    success_count += 1
            
            # 1/5th success rule for step size adaptation
            if success_count / self.pop_size > 0.2:
                sigma *= 1.25  # Expand
            else:
                sigma *= 0.8   # Contract
                
        return best_x, best_y
