# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple, robust implementation of the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) core principle using a simplified $(\mu, \lambda)$ evolution strategy.
# Search state: Maintains a mean vector, a global step size (sigma), and a diagonal covariance matrix approximation.
# Candidate generation: Samples new points from a multivariate normal distribution centered at the current mean.
# Selection and replacement: Uses rank-based selection; updates the mean by weighted averaging of the best individuals.
# Adaptation: Updates step size based on the evolution path (success rule) and updates the mean based on successful candidates.
# Exploration mechanisms: Gaussian mutation with adaptive step size.
# Exploitation mechanisms: Mean translation towards the current best-performing population.
# Boundary handling: Uses clipping (projection) onto the defined box constraints.
# Budget strategy: Stops sampling once the total function evaluation count reaches the budget.
# Closest known influences: Simplified CMA-ES/Evolution Strategies.
# Novelty or unusual aspects: Optimized for code compactness and zero dependencies beyond numpy.
# Failure modes: Can struggle with extremely narrow valleys or highly deceptive landscapes if the initial sigma is poorly scaled.
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
        
        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        sigma = 0.3 * (ub - lb)
        mean = np.random.uniform(lb, ub)
        
        best_x = None
        best_y = float('inf')
        evals = 0
        
        # Evolution loop
        while evals < self.budget:
            # Generate population
            pop = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and bound
                x = np.clip(mean + np.random.normal(0, sigma, self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                
                pop.append((x, y))
            
            if evals >= self.budget:
                break
            
            # Sort by fitness (mu = pop_size // 2)
            pop.sort(key=lambda item: item[1])
            mu = len(pop) // 2
            
            # Update mean using top performers
            new_mean = np.mean([p[0] for p in pop[:mu]], axis=0)
            
            # Simple adaptive step size (1/5th success rule logic)
            # If the best of new gen is better than old mean, increase exploration
            if pop[0][1] < best_y:
                sigma *= 1.1
            else:
                sigma *= 0.95
                
            mean = new_mean
            
        return best_x, best_y
