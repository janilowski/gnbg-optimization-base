# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A lightweight CMA-ES inspired evolution strategy using a diagonal covariance matrix.
# Search state: Maintains a current mean vector and a step-size (sigma).
# Candidate generation: Samples points from a multivariate normal distribution centered at the mean.
# Selection and replacement: Uses (mu, lambda)-selection where the best subset of the population updates the mean.
# Adaptation: Updates the mean using weighted averaging of the best individuals; sigma is adjusted via a simple heuristic.
# Exploration mechanisms: Initialized with a large sigma; stochastic sampling ensures coverage of the search space.
# Exploitation mechanisms: The weighted mean pushes the distribution toward regions of higher fitness.
# Boundary handling: Projects sampled points back into bounds if they exceed allowed limits.
# Budget strategy: Divides budget into generations of a fixed population size until exhausted.
# Closest known influences: CMA-ES, (1+1)-ES, Differential Evolution.
# Novelty or unusual aspects: Minimalist implementation of rank-based evolution without full matrix inversion.
# Failure modes: Can get stuck in local optima if the initial sigma is too small or landscape is highly deceptive.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        self.mu = self.pop_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize state
        x_mean = np.random.uniform(lb, ub, self.dim)
        sigma = 0.3 * (ub - lb)
        best_x = np.copy(x_mean)
        best_y = float('inf')
        
        evals = 0
        while evals < self.budget:
            # Generate population
            pop = []
            fitness = []
            
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and project
                x = x_mean + sigma * np.random.randn(self.dim)
                x = np.clip(x, lb, ub)
                
                y = func(x)
                evals += 1
                
                pop.append(x)
                fitness.append(y)
                
                if y < best_y:
                    best_y = y
                    best_x = np.copy(x)
            
            if not fitness:
                break
                
            # Selection and update
            idx = np.argsort(fitness)
            top_x = np.array([pop[i] for i in idx[:self.mu]])
            
            # Update mean
            new_mean = np.sum(top_x * self.weights[:, np.newaxis], axis=0)
            
            # Adaptive step size simple heuristic
            if np.all(new_mean == x_mean):
                sigma *= 0.5
            else:
                sigma *= 1.05
            
            x_mean = new_mean
            
        return best_x, best_y
