# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy variant.
# Search state: Maintains a mean vector, a covariance matrix (simplified as a diagonal), and a step-size (sigma).
# Candidate generation: Sampling from a multivariate normal distribution centered at the mean.
# Selection and replacement: Fitness-based truncation selection (top 30% of each generation update the mean).
# Adaptation: The mean shifts towards the best candidates; step-size adjusts based on success rate (1/5th rule).
# Exploration mechanisms: Gaussian mutation with adaptive step size.
# Exploitation mechanisms: Adaptive mean shifting towards the current best-performing region.
# Boundary handling: Clamping samples to box constraints.
# Budget strategy: Iteratively generates populations until the function evaluation budget is exhausted.
# Closest known influences: (1+1)-CMA-ES and simple Evolution Strategies.
# Novelty or unusual aspects: Diagonal covariance simplification for robust performance in constrained memory/time.
# Failure modes: Premature convergence in highly non-convex landscapes; performance limited by diagonal assumption.
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

        # Initialization
        evals_left = self.budget
        mean = np.random.uniform(lb, ub)
        sigma = 0.2 * (ub - lb)
        best_x = None
        best_y = float('inf')

        # Population parameters
        pop_size = 4 + int(3 * np.log(self.dim))
        num_parents = max(1, pop_size // 3)

        while evals_left >= pop_size:
            # Generate candidates
            population = [np.clip(mean + np.random.normal(0, sigma), lb, ub) for _ in range(pop_size)]
            
            # Evaluate
            scores = []
            for x in population:
                y = func(x)
                scores.append((y, x))
                if y < best_y:
                    best_y = y
                    best_x = x
                evals_left -= 1
            
            # Selection
            scores.sort(key=lambda x: x[0])
            parents = [s[1] for s in scores[:num_parents]]
            
            # Update mean
            new_mean = np.mean(parents, axis=0)
            
            # Adaptive step size (1/5th rule)
            if scores[0][0] < best_y:
                sigma *= 1.2
            else:
                sigma *= 0.8
                
            mean = new_mean
            
            # Stop if sigma is negligible
            if np.all(sigma < 1e-12):
                break

        # Final cleanup for remaining budget
        while evals_left > 0:
            x = np.clip(mean + np.random.normal(0, sigma * 0.1), lb, ub)
            y = func(x)
            if y < best_y:
                best_y = y
                best_x = x
            evals_left -= 1

        return best_x, best_y
