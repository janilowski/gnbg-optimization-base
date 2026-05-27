# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant utilizing a simplified rank-one update and isotropic step-size control.
# Search state: Maintains a mean vector, a global step size (sigma), and a diagonal covariance factor to track search distribution.
# Candidate generation: Samples candidate solutions from a multivariate normal distribution centered at the mean.
# Selection and replacement: Selects the top 50% of candidates to update the mean vector via weighted averaging.
# Adaptation: Updates step size based on the success rate; adjusts distribution spread based on mean movement.
# Exploration mechanisms: Stochastic sampling governed by the current sigma ensures global search initially.
# Exploitation mechanisms: The mean vector converges toward identified promising regions; local step size reduction refines results.
# Boundary handling: Candidates are clipped to the defined domain bounds before evaluation.
# Budget strategy: Precisely tracks iterations; updates population size dynamically as the budget depletes.
# Closest known influences: CMA-ES and (mu, lambda)-ES.
# Novelty or unusual aspects: Simplified diagonal adaptation for reduced memory overhead and increased speed.
# Failure modes: Can get trapped in narrow local optima in highly deceptive landscapes; may prematurely converge if sigma is too small.
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
        pop_size = int(4 + np.floor(3 * np.log(self.dim)))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)

        x_mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        
        evals = 0
        while evals < self.budget:
            # Generate candidates
            candidates = []
            fitness = []
            
            for _ in range(pop_size):
                if evals >= self.budget: break
                
                # Sample and bound
                x = np.clip(np.random.normal(x_mean, sigma), lb, ub)
                y = func(x)
                evals += 1
                
                candidates.append(x)
                fitness.append(y)
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            
            if not candidates: break
            
            # Selection
            idx = np.argsort(fitness)
            top_indices = idx[:mu]
            parents = np.array([candidates[i] for i in top_indices])
            
            # Adaptation
            old_mean = x_mean.copy()
            x_mean = np.dot(weights, parents)
            
            # Step size adaptation: Increase if mean moves, decrease if stagnant
            if np.linalg.norm(x_mean - old_mean) > np.mean(sigma):
                sigma *= 1.05
            else:
                sigma *= 0.95
                
            sigma = np.clip(sigma, 1e-10 * (ub - lb), 1.0 * (ub - lb))
            
        return best_x, best_y
