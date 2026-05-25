# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact implementation of the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) derivative using a simplified rank-one update strategy.
# Search state: Maintains a mean vector, a global step size (sigma), and a diagonal covariance approximation.
# Candidate generation: Samples candidate solutions from a multivariate normal distribution centered at the mean with a scaled diagonal covariance.
# Selection and replacement: Uses a (mu, lambda) selection scheme, updating the mean based on the top-performing fraction of the population.
# Adaptation: Updates sigma via cumulative step-size adaptation and adjusts the mean using a weighted average of the best samples.
# Exploration mechanisms: The initial large sigma combined with stochastic sampling ensures broad coverage, while the evolution path encourages global directional movement.
# Exploitation mechanisms: The mean vector converges toward the local optimum, while the covariance adaptation refines search step sizes along principal dimensions.
# Boundary handling: Uses a simple mirror/resampling strategy where candidates outside bounds are projected back to the feasible region.
# Budget strategy: Employs a generation-based loop that terminates immediately when the evaluation count reaches the budget.
# Closest known influences: CMA-ES and simple Evolution Strategies (ES).
# Novelty or unusual aspects: Minimalist diagonal-only covariance structure for memory efficiency and robustness in higher dimensions.
# Failure modes: Susceptible to premature convergence if the initial sigma is poorly scaled relative to the landscape topology.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Strategy parameters
        pop_size = 4 + int(3 * np.log(self.dim))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')

        while self.evals < self.budget:
            # Generate candidates
            candidates = []
            for _ in range(pop_size):
                if self.evals >= self.budget:
                    break
                sample = mean + sigma * np.random.randn(self.dim)
                # Boundary projection
                sample = np.clip(sample, lb, ub)
                val = func(sample)
                self.evals += 1
                
                if val < best_y:
                    best_y = val
                    best_x = sample.copy()
                candidates.append((val, sample))
            
            # Selection
            candidates.sort(key=lambda x: x[0])
            top_samples = np.array([c[1] for c in candidates[:mu]])
            
            # Update mean
            new_mean = np.dot(weights, top_samples)
            
            # Update sigma (adaptive step size)
            sigma = sigma * 0.95 + 0.05 * np.std(top_samples, axis=0).mean()
            
            # Move mean
            mean = new_mean
            
        return best_x, best_y
