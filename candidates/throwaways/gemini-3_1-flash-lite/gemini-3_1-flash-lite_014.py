# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Simple Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant.
# Search state: Maintains a mean vector, a global step size (sigma), and an identity covariance matrix (simplified CMA).
# Candidate generation: Samples new points from a multivariate normal distribution defined by the current mean and sigma.
# Selection and replacement: Selects the best elite fraction of the population to compute a weighted mean update.
# Adaptation: Updates mean towards the successful elite, and adjusts sigma based on the success rate.
# Exploration mechanisms: Global search is driven by the initial large sigma and Gaussian sampling.
# Exploitation mechanisms: Local search is driven by the shrinking sigma and iterative centering on the best observed solution.
# Boundary handling: Implements hard projection (clamping) to the defined problem bounds.
# Budget strategy: Divides the budget into generations, ceasing when the evaluation count reaches zero.
# Closest known influences: Simplified (1+1)-ES and basic CMA-ES logic.
# Novelty or unusual aspects: Uses a simplified covariance update to keep the implementation standard library compliant.
# Failure modes: Can get stuck in local optima if the initial population does not capture the basin of attraction.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        self.elite_size = self.pop_size // 2

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower'):
            lb, ub = func.lower, func.upper
        else:
            lb, ub = func.bounds.lb, func.bounds.ub
            
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Initialize state
        curr_mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        evals_left = self.budget

        while evals_left >= self.pop_size:
            # Generate population
            pop = []
            for _ in range(self.pop_size):
                # Clamp sample to bounds
                x = np.clip(curr_mean + sigma * np.random.randn(self.dim), lb, ub)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x
                pop.append((x, y))
                evals_left -= 1
            
            # Selection
            pop.sort(key=lambda item: item[1])
            elites = pop[:self.elite_size]
            
            # Update mean
            new_mean = np.mean([p[0] for p in elites], axis=0)
            
            # Adaptation: adjust sigma based on movement
            dist = np.linalg.norm(new_mean - curr_mean)
            if dist > 0:
                sigma *= 1.05
            else:
                sigma *= 0.95
                
            curr_mean = new_mean
            
            # Safety check: keep sigma within bounds
            sigma = np.clip(sigma, 1e-6 * (ub - lb), 0.5 * (ub - lb))

        return best_x, best_y
