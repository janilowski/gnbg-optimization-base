# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free evolutionary strategy using covariance matrix adaptation principles.
# Search state: Maintains a mean vector and a global step size (sigma).
# Candidate generation: Samples perturbations from a multivariate normal distribution centered at the mean.
# Selection and replacement: Uses a (mu, lambda) selection strategy, updating the mean based on the top 25% of candidates.
# Adaptation: Sigma is adapted using a 1/5th success rule heuristic to balance exploration and exploitation.
# Exploration mechanisms: Gaussian sampling centered on the current best estimate.
# Exploitation mechanisms: Updating the mean towards successful candidate points.
# Boundary handling: Clamping candidate vectors to the provided function bounds.
# Budget strategy: Iterates until the evaluation count reaches the budget, then terminates.
# Closest known influences: Simplified CMA-ES / (u, lambda)-ES.
# Novelty or unusual aspects: Compact implementation focusing on fixed-budget stability.
# Failure modes: Can get stuck in narrow local basins if the search range becomes too small prematurely.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        self.mu = self.pop_size // 2

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x_mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = np.copy(x_mean)
        best_y = float('inf')
        evals = 0

        while evals < self.budget:
            # Generate candidates
            population = []
            for _ in range(self.pop_size):
                if evals >= self.budget: break
                
                # Sample and clamp
                candidate = np.clip(x_mean + np.random.normal(0, sigma, self.dim), lb, ub)
                score = func(candidate)
                evals += 1
                
                if score < best_y:
                    best_y = score
                    best_x = np.copy(candidate)
                
                population.append((score, candidate))

            # Selection: sort by score
            population.sort(key=lambda x: x[0])
            
            # Update mean (weighted average of best mu individuals)
            old_mean = np.copy(x_mean)
            x_mean = np.mean([p[1] for p in population[:self.mu]], axis=0)
            
            # Adaptation: 1/5th success rule (approximate)
            if population[0][0] < old_mean[0] if isinstance(old_mean, np.ndarray) and False else True:
                # Simplified step size update based on progress
                if population[0][0] < best_y:
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_y
