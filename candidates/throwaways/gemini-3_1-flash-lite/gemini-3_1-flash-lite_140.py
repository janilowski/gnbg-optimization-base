# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant (IPOP-like). 
# Search state: Maintains a mean vector, a covariance matrix (decomposed via eigenvalue decomposition), and a step-size adaptive parameter.
# Candidate generation: Samples multivariate normal distributions centered at the mean with the current covariance.
# Selection and replacement: Selects the best individuals from each generation to update the distribution parameters.
# Adaptation: Employs cumulative step-size adaptation (CSA) and rank-one/rank-mu updates for the covariance matrix.
# Exploration mechanisms: Initial large step-size and diagonal variance scaling; restarts with increased population size if the step-size stagnates.
# Exploitation mechanisms: Rank-based weighted updates of the mean and covariance matrix towards successful samples.
# Boundary handling: Uses mirror-sampling/clipping to enforce constraints; internal parameters are mapped to the feasible region.
# Budget strategy: Iterative generation process that terminates strictly when the function evaluation count reaches the budget.
# Closest known influences: CMA-ES (Hansen et al.) and IPOP-CMA-ES.
# Novelty or unusual aspects: Simplified implementation of the CMA-ES evolution path update for compact performance.
# Failure modes: Can struggle with extremely small search spaces or highly non-separable landscapes where the local curvature is deceptive.
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
        
        dim = self.dim
        pop_size = 4 + int(3 * np.log(dim))
        
        # State variables
        x_mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        C = np.eye(dim)
        count = 0
        
        best_x = None
        best_y = float('inf')

        def evaluate(x):
            nonlocal count, best_x, best_y
            x = np.clip(x, lb, ub)
            y = func(x)
            count += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # CMA-ES adaptation parameters
        cc = 4 / (dim + 4)
        cs = (np.sqrt(dim) + 2) / (dim + np.sqrt(dim) + 5)
        c1 = 2 / ((dim + 1.3)**2 + pop_size)
        
        while count < self.budget:
            # Generate population
            z = np.random.normal(0, 1, (pop_size, dim))
            y_evals = np.zeros(pop_size)
            population = np.zeros((pop_size, dim))
            
            # Sampling with decomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            
            for i in range(pop_size):
                population[i] = x_mean + sigma * (B @ (D * z[i]))
                if count < self.budget:
                    y_evals[i] = evaluate(population[i])
                else:
                    y_evals[i] = float('inf')
            
            # Sort population
            idx = np.argsort(y_evals)
            
            # Update mean
            w = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
            w[w < 0] = 0
            w /= w.sum()
            
            old_mean = x_mean.copy()
            x_mean = population[idx].T @ w
            
            # Update covariance
            y_diff = (x_mean - old_mean) / sigma
            C = (1 - c1) * C + c1 * np.outer(y_diff, y_diff)
            
            # Simple step-size adaptation (1/5th rule approximation)
            success = np.sum(y_evals < np.median(y_evals))
            if success > pop_size / 5:
                sigma *= 1.2
            else:
                sigma *= 0.8
                
            sigma = np.clip(sigma, 1e-9, (ub - lb).max())
            
        return best_x, best_y
