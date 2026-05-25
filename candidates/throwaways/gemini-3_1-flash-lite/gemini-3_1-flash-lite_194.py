# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) core logic for black-box optimization.
# Search state: Maintains a mean vector, a step-size (sigma), and an identity covariance matrix for simplicity/low-dim robustness.
# Candidate generation: Multivariate normal sampling around the mean.
# Selection and replacement: Fitness-based truncation selection (top 50% of the population).
# Adaptation: Updates the mean toward the best individuals; maintains a fixed sigma for steady exploration.
# Exploration mechanisms: Gaussian noise sampling with adaptive scaling.
# Exploitation mechanisms: Weighted mean shift towards successful candidates.
# Boundary handling: Clipping values to defined bounds during candidate evaluation.
# Budget strategy: Iterative loop that terminates exactly when the evaluation count reaches the budget.
# Closest known influences: Simplified (1+1)-ES and basic CMA-ES principles.
# Novelty or unusual aspects: Simplified variance tracking to ensure robust performance without complex matrix decompositions.
# Failure modes: Can get trapped in narrow local minima; requires sufficient budget for initial exploration.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialize parameters
        pop_size = 4 + int(3 * np.log(self.dim))
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        evals = 0

        while evals < self.budget:
            # Generate candidates
            candidates = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and project to bounds
                x = np.clip(mean + sigma * np.random.randn(self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x
                
                candidates.append((x, y))
            
            # Sort by fitness
            candidates.sort(key=lambda item: item[1])
            
            # Update mean using the best half
            num_elite = max(1, pop_size // 2)
            elite = [c[0] for c in candidates[:num_elite]]
            mean = np.mean(elite, axis=0)
            
            # Decay sigma slightly to sharpen exploitation over time
            sigma *= 0.99
            
            # If search is stuck, re-randomize mean to escape
            if np.all(sigma < 1e-5 * (ub - lb)):
                mean = np.random.uniform(lb, ub)
                sigma = 0.3 * (ub - lb)

        return best_x, best_y
