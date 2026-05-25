# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES-inspired evolution strategy using a simplified covariance matrix adaptation.
# Search state: Maintains a mean vector, a global step size (sigma), and a diagonal covariance approximation.
# Candidate generation: Samples new points from a multivariate normal distribution centered at the mean.
# Selection and replacement: Uses a (mu, lambda) strategy, selecting the best fraction of candidates to update the mean.
# Adaptation: Updates step size based on success rate and adjusts the mean using a weighted average of successful samples.
# Exploration mechanisms: Initialized with a large sigma; variance is controlled via an empirical success rule.
# Exploitation mechanisms: The mean shifts toward the best observed points, refining the locality of the search.
# Boundary handling: Samples are clipped to the function bounds to ensure feasibility.
# Budget strategy: Precisely manages evaluations by splitting the budget into discrete generations.
# Closest known influences: CMA-ES, 1+1 Evolution Strategy.
# Novelty or unusual aspects: Simplified diagonal covariance update to minimize computation and memory overhead.
# Failure modes: Can get trapped in sharp local minima if parameters are poorly scaled; susceptible to stagnation in high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = func.lower, func.upper
        else:
            lb, ub = func.bounds.lb, func.bounds.ub

        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        num_parents = pop_size // 2
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.2 * (ub - lb)
        best_x = None
        best_y = float('inf')
        evals_remaining = self.budget

        while evals_remaining > 0:
            # Generate offspring
            candidates = []
            for _ in range(min(pop_size, evals_remaining)):
                z = np.random.normal(0, 1, self.dim)
                x = np.clip(mean + sigma * z, lb, ub)
                y = func(x)
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                
                candidates.append((x, y))
                evals_remaining -= 1
            
            # Sort candidates by fitness
            candidates.sort(key=lambda item: item[1])
            
            # Update mean using top parents
            parents = [c[0] for c in candidates[:num_parents]]
            new_mean = np.mean(parents, axis=0)
            
            # Step size adaptation: simple success rule
            if np.mean([c[1] for c in candidates[:num_parents]]) < best_y:
                sigma *= 1.05
            else:
                sigma *= 0.95
                
            mean = new_mean
            
            # Early break if sigma is effectively zero
            if np.all(sigma < 1e-10):
                break
                
        return best_x, best_y
