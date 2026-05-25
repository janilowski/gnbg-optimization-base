# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free covariance matrix adaptation evolution strategy (CMA-ES) variant utilizing a simplified CMA approach.
# Search state: Maintains a mean vector, a global step size (sigma), and an identity-based covariance matrix.
# Candidate generation: Samples new points from a multivariate normal distribution centered at the current mean.
# Selection and replacement: Uses rank-based selection where the best half of the population updates the mean.
# Adaptation: Updates the mean using weighted recombination; step size is adjusted via the success rule (1/5th rule).
# Exploration mechanisms: Gaussian noise controlled by the step size sigma.
# Exploitation mechanisms: Moving the distribution mean toward the best-performing samples.
# Boundary handling: Clamping sampled points to the provided box constraints.
# Budget strategy: Iteratively evaluates population chunks until the budget is exhausted.
# Closest known influences: (1+1)-ES and simplified CMA-ES.
# Novelty or unusual aspects: Compact implementation focusing on rapid convergence in low-to-medium dimensions.
# Failure modes: Slow convergence on highly non-convex or needle-in-a-haystack landscapes; step size may collapse prematurely.
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

        # Initialize state
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        evals = 0

        while evals < self.budget:
            # Generate population
            candidates = []
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and clamp
                x = np.clip(mean + sigma * np.random.randn(self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                candidates.append((x, y))
                if y < best_y:
                    best_y = y
                    best_x = x

            # Sort population by objective value
            candidates.sort(key=lambda item: item[1])
            
            # Select best half for mean update
            keep = max(1, len(candidates) // 2)
            new_mean = np.mean([c[0] for c in candidates[:keep]], axis=0)
            
            # Update step size based on drift (1/5th success rule)
            if candidates[0][1] < best_y:
                sigma *= 1.2
            else:
                sigma *= 0.8
            
            mean = new_mean
            
        return best_x, best_y
