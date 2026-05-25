# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution 
#          Strategy (CMA-ES) inspired simplified approach.
# Search state: Maintains a current mean vector and a step size (sigma).
# Candidate generation: Generates multivariate normal distributed samples around the mean.
# Selection and replacement: Uses rank-based selection (elitism) to update the mean.
# Adaptation: Updates sigma based on successful step sizes; the mean vector evolves towards 
#             the best performing candidates.
# Exploration mechanisms: Large initial sigma and stochastic sampling provide global search.
# Exploitation mechanisms: The mean shifts towards better candidates; local refinement as sigma decays.
# Boundary handling: Hard clipping ensures all candidates remain within the specified search bounds.
# Budget strategy: Uniform distribution of evaluations over a fixed number of generations.
# Closest known influences: Simplified (1+1)-CMA-ES and basic local evolution strategies.
# Novelty or unusual aspects: Minimalist implementation of rank-based evolution without heavy matrix math.
# Failure modes: Can get stuck in local optima if the initial population variance is too small.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.max_gens = 100
        self.pop_size = max(4, int(np.sqrt(self.dim) * 4))
        self.evals_per_gen = self.pop_size
        self.num_gens = min(self.max_gens, budget // self.pop_size)

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        else:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)

        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = np.copy(mean)
        best_y = float('inf')
        eval_count = 0

        for gen in range(self.num_gens):
            # Generate population
            samples = mean + sigma * np.random.normal(0, 1, (self.pop_size, self.dim))
            # Clip to bounds
            samples = np.clip(samples, lb, ub)

            # Evaluate
            scores = np.array([func(x) for x in samples])
            eval_count += self.pop_size

            # Update best
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_y:
                best_y = scores[min_idx]
                best_x = np.copy(samples[min_idx])

            # Selection: Use top 50% to update mean
            indices = np.argsort(scores)
            elite_indices = indices[:self.pop_size // 2]
            mean = np.mean(samples[elite_indices], axis=0)

            # Adaptive step size: shrink as search progresses
            sigma *= 0.95

            if eval_count >= self.budget:
                break

        return best_x, best_y
