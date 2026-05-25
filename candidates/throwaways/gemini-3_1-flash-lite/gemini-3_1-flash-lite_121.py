# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES-inspired evolution strategy simplified for black-box optimization.
# Search state: Maintains a Gaussian distribution defined by a mean vector and global step size (sigma).
# Candidate generation: Samples N candidates from a multivariate normal distribution centered at the current mean.
# Selection and replacement: Selects the top 50% of candidates and updates the mean to the weighted average of these elites.
# Adaptation: Updates sigma based on the success rate; if progress is made, the search radius expands, otherwise it contracts.
# Exploration mechanisms: Initialized with a large sigma; stochastic sampling ensures coverage of the search space.
# Exploitation mechanisms: The mean gradually converges toward the local optimum (centroid shift).
# Boundary handling: Candidates are clipped to the provided bounds, and out-of-bounds samples are re-sampled.
# Budget strategy: A fixed number of iterations is computed based on the budget and population size.
# Closest known influences: (1+1)-ES and simple CMA-ES variants.
# Novelty or unusual aspects: Uses a simple adaptive step-size mechanism without full covariance matrix adaptation.
# Failure modes: Susceptible to local minima in highly multimodal functions; performance degrades if the landscape is very sparse.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(4 + int(3 * np.log(dim)), budget)
        self.max_iter = budget // self.pop_size

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        mean = np.random.uniform(lb, ub, self.dim)
        sigma = 0.2 * (ub - lb)
        best_x = None
        best_y = float('inf')
        
        evals = 0
        for _ in range(self.max_iter):
            if evals >= self.budget:
                break
            
            # Generate candidates
            candidates = []
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and enforce bounds
                x = np.clip(mean + np.random.normal(0, sigma, self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                candidates.append((x, y))
            
            # Selection: Sort by fitness
            candidates.sort(key=lambda item: item[1])
            elites = candidates[:max(1, len(candidates) // 2)]
            
            # Update mean
            new_mean = np.mean([c[0] for c in elites], axis=0)
            
            # Adaptation: Simple step-size update
            if new_mean is not None:
                diff = new_mean - mean
                mean = new_mean
                # Expand if moving, contract if stuck
                if np.linalg.norm(diff) > 1e-9:
                    sigma *= 1.1
                else:
                    sigma *= 0.8
            
            sigma = np.clip(sigma, 1e-6, 0.5 * (ub - lb))
            
        return best_x, best_y
