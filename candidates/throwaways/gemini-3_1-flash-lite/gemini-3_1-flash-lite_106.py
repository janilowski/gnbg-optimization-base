# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES-inspired evolution strategy using a diagonal covariance matrix.
# Search state: Maintains a mean vector, a diagonal covariance vector (step sizes), and a learning rate.
# Candidate generation: Samples candidates from a multivariate normal distribution centered at the current mean.
# Selection and replacement: Uses a (mu, lambda) selection strategy, updating the mean based on the top 30% of candidates.
# Adaptation: Updates step sizes using a simplified adaptive scheme based on successful moves.
# Exploration mechanisms: Initialized with a large variance; uses normal distribution sampling to cover the search space.
# Exploitation mechanisms: Mean moves towards the best sampled points; variance shrinks as the search converges.
# Boundary handling: candidates are projected (clamped) back into the defined search space bounds.
# Budget strategy: Precisely tracks evaluations until the budget is exhausted.
# Closest known influences: (1+1)-ES and simple CMA-ES variants.
# Novelty or unusual aspects: Extremely compact implementation using only standard library and numpy.
# Failure modes: Susceptible to local optima in highly multimodal landscapes or if the initial step size is poorly tuned.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = (ub - lb) * 0.3
        pop_size = 4 + int(3 * np.log(self.dim))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        best_x = None
        best_y = float('inf')
        
        while self.evals < self.budget:
            # Generate population
            candidates = []
            for _ in range(pop_size):
                if self.evals >= self.budget:
                    break
                x = np.clip(mean + sigma * np.random.randn(self.dim), lb, ub)
                y = func(x)
                self.evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                candidates.append((x, y))
            
            # Sort and update mean
            candidates.sort(key=lambda item: item[1])
            best_candidates = candidates[:mu]
            
            # Update mean using top performers
            new_mean = np.zeros(self.dim)
            for i in range(mu):
                new_mean += weights[i] * best_candidates[i][0]
            
            # Adaptation of sigma (simple step-size control)
            diff = (new_mean - mean)
            mean = new_mean
            sigma = np.clip(sigma * 0.9 + np.abs(diff) * 0.1, (ub - lb) * 1e-5, (ub - lb) * 0.5)
            
        return best_x, best_y
