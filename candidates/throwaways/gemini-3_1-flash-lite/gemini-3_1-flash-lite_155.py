# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A population-based (mu+lambda) evolutionary strategy with covariance matrix adaptation (simplified CMA-ES style).
# Search state: Maintains a current mean vector and a step-size (sigma), updating them based on successful iterations.
# Candidate generation: Samples new points from a multivariate normal distribution centered at the mean.
# Selection and replacement: Uses (mu, lambda) selection, keeping the best individuals to compute the next mean.
# Adaptation: Updates sigma using the success rule (1/5th rule) and drifts the mean towards the best candidates.
# Exploration mechanisms: Gaussian mutation with adaptive step size.
# Exploitation mechanisms: Mean displacement and rank-based selection.
# Boundary handling: Resampling or clipping points that exceed bounds.
# Budget strategy: Exhausts the budget by running generations until the total evaluations hit the limit.
# Closest known influences: Simplified (1+1)-ES and CMA-ES variants.
# Novelty or unusual aspects: Simplified variance control without full covariance matrix reconstruction.
# Failure modes: Can get stuck in local optima if the initial sigma is too small or landscape is highly deceptive.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.2 * (ub - lb)
        eval_count = 0
        best_x = None
        best_y = float('inf')

        # Population parameters
        pop_size = 4 + int(3 * np.log(self.dim))
        
        while eval_count < self.budget:
            # Generate candidates
            population = []
            for _ in range(pop_size):
                if eval_count >= self.budget:
                    break
                
                # Sample and clip
                x = np.clip(mean + sigma * np.random.normal(0, 1, self.dim), lb, ub)
                y = func(x)
                eval_count += 1
                
                population.append((x, y))
                
                if y < best_y:
                    best_y = y
                    best_x = x

            # Sort population
            population.sort(key=lambda item: item[1])
            
            # Selection: Update mean based on top half
            mu = pop_size // 2
            new_mean = np.mean([p[0] for p in population[:mu]], axis=0)
            
            # Adaptation: 1/5th rule inspired step size update
            # If the best improved overall, expand; else shrink
            if population[0][1] < best_y:
                sigma *= 1.2
            else:
                sigma *= 0.8
            
            mean = new_mean
            
            # Prevent stagnation
            if np.all(sigma < 1e-10):
                sigma = 0.1 * (ub - lb)

        return best_x, best_y
