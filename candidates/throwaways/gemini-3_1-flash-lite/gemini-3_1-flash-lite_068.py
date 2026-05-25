# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (GNBG) trust-region-inspired CMA-ES variation using a covariance matrix adaptation mechanism with a reduced population size for dimensionality efficiency.
# Search state: Maintains a current mean vector, a covariance matrix (simplified as diagonal), and a step-size (sigma).
# Candidate generation: Generates samples from a multivariate normal distribution centered at the current mean, scaled by sigma.
# Selection and replacement: Selects the best performing samples to update the mean via a weighted average.
# Adaptation: Updates sigma using the success-rate rule (1/5th success rule) and adapts the diagonal covariance based on successful step directions.
# Exploration mechanisms: Gaussian noise controlled by sigma ensures global search, while the covariance adaptation tunes the search landscape.
# Exploitation mechanisms: Mean shift toward better performing regions and shrinking sigma as progress stagnates.
# Boundary handling: Resampling or projection to the feasible region defined by the objective function's bounds.
# Budget strategy: Static population size and strict termination once the iteration count exhausts the provided budget.
# Closest known influences: Inspired by CMA-ES and simple Evolution Strategies (1+lambda).
# Novelty or unusual aspects: Employs a lightweight diagonal covariance adaptation to keep memory and computation low for high dimensions.
# Failure modes: Can get stuck in narrow local minima if the initial sigma is too small or if the objective landscape is extremely deceptive.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        
    def __call__(self, func):
        # Extract bounds
        lb = getattr(func, 'lower', None)
        if lb is None:
            lb = func.bounds.lb
        ub = getattr(func, 'upper', None)
        if ub is None:
            ub = func.bounds.ub
        
        lb, ub = np.array(lb), np.array(ub)
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        diag_cov = np.ones(self.dim)
        
        evals = 0
        best_x = None
        best_y = float('inf')
        
        while evals < self.budget:
            # Generate population
            samples = []
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and project to bounds
                z = np.random.normal(0, 1, self.dim)
                x = mean + sigma * (z * np.sqrt(diag_cov))
                x = np.clip(x, lb, ub)
                
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                samples.append((x, y))
            
            # Selection: Sort by fitness
            samples.sort(key=lambda s: s[1])
            
            # Adaptation: Move mean toward the best sample (weighted)
            weights = np.log(self.pop_size + 0.5) - np.log(np.arange(1, self.pop_size + 1))
            weights /= weights.sum()
            
            old_mean = mean.copy()
            mean = sum(weights[i] * samples[i][0] for i in range(len(samples)))
            
            # Adaptation: Update step-size (1/5th rule heuristic)
            if samples[0][1] < best_y:
                sigma *= 1.1
            else:
                sigma *= 0.95
                
            # Adaptation: Diagonal covariance update
            diff = (mean - old_mean) / (sigma + 1e-9)
            diag_cov = 0.9 * diag_cov + 0.1 * (diff**2)
            
        return best_x, best_y
