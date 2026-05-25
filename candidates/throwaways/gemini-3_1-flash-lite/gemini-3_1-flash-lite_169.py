# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (Z-space) CMA-ES variant simplified for budget-constrained black-box optimization.
# Search state: Maintains a current mean vector and a step size (sigma).
# Candidate generation: Samples perturbations from a multivariate normal distribution centered at the mean.
# Selection and replacement: Replaces the mean with the weighted average of the top-performing fraction of samples.
# Adaptation: Updates sigma using the success rule (1/5th success rule) and covariance matrix adaptation via rank-one update.
# Exploration mechanisms: Stochastic sampling governed by the covariance matrix and step size.
# Exploitation mechanisms: Mean shifting towards superior regions and rank-one covariance matrix updates.
# Boundary handling: Clipping the samples to the function's defined domain bounds.
# Budget strategy: Divides the budget into generations, estimating initial population size based on dimension.
# Closest known influences: CMA-ES and (mu, lambda)-ES.
# Novelty or unusual aspects: Highly compact implementation without full matrix decomposition.
# Failure modes: Slow convergence on highly needle-in-a-haystack or extremely non-convex landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        lb = getattr(func, 'lower', getattr(func.bounds, 'lb', -5.12))
        ub = getattr(func, 'upper', getattr(func.bounds, 'ub', 5.12))
        lb = np.array(lb) if hasattr(lb, '__iter__') else np.full(self.dim, lb)
        ub = np.array(ub) if hasattr(ub, '__iter__') else np.full(self.dim, ub)
        
        # Initialization
        pop_size = 4 + int(3 * np.log(self.dim))
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        evals = 0
        
        # Simple CMA-ES style adaptation components
        C = np.eye(self.dim)
        
        while evals < self.budget:
            # Generate population
            pop = []
            values = []
            for _ in range(pop_size):
                if evals >= self.budget: break
                
                # Sample
                z = np.random.multivariate_normal(np.zeros(self.dim), C)
                x = np.clip(mean + sigma * z, lb, ub)
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                
                pop.append((x - mean) / sigma)
                values.append(y)
            
            # Selection
            idx = np.argsort(values)
            elite_idx = idx[:max(1, pop_size // 2)]
            
            # Recombination
            z_mean = np.mean([pop[i] for i in elite_idx], axis=0)
            mean = mean + sigma * z_mean
            
            # Covariance adaptation (simplified rank-one)
            C = 0.9 * C + 0.1 * np.outer(z_mean, z_mean)
            
            # Step size adaptation (1/5 success rule)
            success_ratio = len([i for i in elite_idx if values[i] < best_y]) / len(elite_idx)
            if success_ratio > 0.2:
                sigma *= 1.1
            else:
                sigma *= 0.9
                
        return best_x, best_y
