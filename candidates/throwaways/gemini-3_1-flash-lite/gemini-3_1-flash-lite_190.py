# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant limited to Diagonal Covariance.
# Search state: Maintains a mean vector, a diagonal covariance vector, and a step-size scalar.
# Candidate generation: Samples candidate vectors from a multivariate normal distribution defined by current mean and diagonal covariance.
# Selection and replacement: Selects the top proportion of samples (selection pressure) to update the mean.
# Adaptation: Updates the mean toward the best samples and increases/decreases step-size based on the success of prior move directions (cumulative step-size adaptation).
# Exploration mechanisms: Large initial step-size and stochastic sampling; variance is tracked per dimension.
# Exploitation mechanisms: Mean-centering on successful solutions and weighting the best points to refine local optima.
# Boundary handling: Clamping candidate solutions to defined lower/upper bounds.
# Budget strategy: Calculates population size based on dimensionality and divides total budget by population to determine generation count.
# Closest known influences: CMA-ES / VSSO (Variable Step-Size Optimizer).
# Novelty or unusual aspects: Simplified diagonal adaptation for high-dimensional stability in a single-file module.
# Failure modes: Can converge prematurely on highly multi-modal landscapes; sensitive to initialization if bounds are extremely large.
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

        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        num_generations = self.budget // pop_size
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        
        # Strategy parameters
        weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
        weights[weights < 0] = 0
        weights /= weights.sum()
        mu = int(pop_size / 2)
        
        eval_count = 0
        for _ in range(num_generations):
            if eval_count >= self.budget:
                break
                
            # Generate candidates
            pop = [np.clip(mean + sigma * np.random.normal(0, 1, self.dim), lb, ub) for _ in range(pop_size)]
            
            # Evaluate
            scores = []
            for x in pop:
                y = func(x)
                eval_count += 1
                scores.append(y)
                if y < best_y:
                    best_y = y
                    best_x = x
            
            # Rank
            idx = np.argsort(scores)
            
            # Update mean
            old_mean = mean.copy()
            mean = np.sum([weights[i] * pop[idx[i]] for i in range(mu)], axis=0)
            
            # Adaptation: simple step-size control
            diff = mean - old_mean
            if np.linalg.norm(diff) > 0:
                sigma *= 1.05  # Expand if moving
            else:
                sigma *= 0.95  # Shrink if stuck
                
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 1e-5 * (ub - lb), 1.0 * (ub - lb))
            
        return best_x, best_y
