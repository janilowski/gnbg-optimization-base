# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free covariance matrix adaptation evolution strategy variant (simplified CMA-ES).
# Search state: Maintains a mean vector, a step-size (sigma), and an identity-based covariance matrix.
# Candidate generation: Samples individuals from a multivariate normal distribution centered at the mean.
# Selection and replacement: Uses rank-based selection on the population; updates the mean towards the best candidates.
# Adaptation: Integrates a basic step-size control and updates the mean based on the top 50% of the population.
# Exploration mechanisms: Initialized with a large sigma covering the search space, scaling down over time.
# Exploitation mechanisms: Moves the mean vector toward the weighted average of the fittest candidates collected per generation.
# Boundary handling: Uses clipping (projection) to keep candidates within the specified function bounds.
# Budget strategy: Divides the total budget into generations of fixed size (lambda = 4 + floor(3*ln(dim))) until exhausted.
# Closest known influences: Simplified CMA-ES and (mu, lambda)-ES.
# Novelty or unusual aspects: Extremely compact implementation of a rank-based evolutionary adaptive strategy.
# Failure modes: May get stuck in local optima if the initial sigma is too small or if the landscape is highly deceptive.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        num_parents = pop_size // 2
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        
        while self.evals + pop_size <= self.budget:
            # Generate candidates
            population = [np.clip(mean + sigma * np.random.randn(self.dim), lb, ub) for _ in range(pop_size)]
            
            # Evaluate
            fitness = []
            for x in population:
                y = func(x)
                self.evals += 1
                fitness.append((y, x))
                if y < best_y:
                    best_y = y
                    best_x = x
            
            # Selection: Sort by fitness
            fitness.sort(key=lambda item: item[0])
            parents = [item[1] for item in fitness[:num_parents]]
            
            # Adaptation: Move mean towards parents
            new_mean = np.mean(parents, axis=0)
            
            # Simple step-size adaptation (decay)
            # If population variance is low, shrink sigma
            if np.linalg.norm(new_mean - mean) < 1e-4 * np.mean(sigma):
                sigma *= 0.5
            
            mean = new_mean
            
            # Safety break for stagnating sigma
            if np.all(sigma < 1e-10):
                sigma = 0.1 * (ub - lb)
                
        # Final cleanup for remaining budget
        while self.evals < self.budget:
            x = np.clip(best_x + 0.1 * sigma * np.random.randn(self.dim), lb, ub)
            y = func(x)
            self.evals += 1
            if y < best_y:
                best_y, best_x = y, x

        return best_x, best_y
