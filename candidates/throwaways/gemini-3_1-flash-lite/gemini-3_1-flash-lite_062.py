# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant with a simplified diagonal approximation.
# Search state: Maintains a mean vector, a global step size (sigma), and a diagonal covariance vector to track search distribution.
# Candidate generation: Samples candidate solutions from a multivariate normal distribution scaled by the current step size.
# Selection and replacement: Uses rank-based selection, updating the mean towards the best candidates (weighted recombination).
# Adaptation: Updates the mean vector and adjusts the step size and covariance diagonal based on successful progress.
# Exploration mechanisms: Initialized with a large sigma; stochastic sampling ensures coverage of the search space.
# Exploitation mechanisms: The mean vector shifts towards optimal regions; step size shrinks as the population converges.
# Boundary handling: Projects candidates back into the [lb, ub] box using clipping if they exceed bounds.
# Budget strategy: Employs a fixed population size proportional to dimensionality; terminates once evaluation budget is exhausted.
# Closest known influences: Simplified CMA-ES / Variable Metric Evolution Strategy.
# Novelty or unusual aspects: Uses a lightweight diagonal scaling approach to reduce computation while maintaining adaptive capabilities.
# Failure modes: Can get trapped in local optima for highly multi-modal landscapes; might struggle with extremely narrow, deep ridges.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        try:
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        except AttributeError:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)

        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        diag_c = np.ones(self.dim)
        
        best_x = None
        best_y = float('inf')
        evals = 0

        while evals < self.budget:
            # Generate offspring
            samples = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and project
                z = np.random.normal(0, 1, self.dim)
                x = mean + sigma * (diag_c * z)
                x = np.clip(x, lb, ub)
                
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = np.copy(x)
                
                samples.append((y, x))
            
            # Sort by fitness (simple selection)
            samples.sort(key=lambda item: item[0])
            
            # Recombination: update mean toward the best half of the population
            num_keep = max(1, pop_size // 2)
            parents = [s[1] for s in samples[:num_keep]]
            new_mean = np.mean(parents, axis=0)
            
            # Adaptation: update "step" direction
            step = new_mean - mean
            mean = new_mean
            
            # Heuristic adaptation of sigma
            if np.linalg.norm(step) > 0:
                sigma *= 1.05  # Success: expand
            else:
                sigma *= 0.95  # Stagnation: contract
                
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 1e-4 * (ub - lb), 1.0 * (ub - lb))
            
        return best_x, best_y
