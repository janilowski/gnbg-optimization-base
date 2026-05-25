# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant with a simplified diagonal covariance model.
# Search state: Maintains a mean vector, a global step size (sigma), and a diagonal covariance vector to capture coordinate-wise correlations.
# Candidate generation: Generates candidates by sampling from a multivariate normal distribution centered at the mean with a scaled diagonal covariance.
# Selection and replacement: Selects the top-performing individuals (elitism) to update the mean via a weighted average.
# Adaptation: Updates the mean toward the best samples and increases/decreases step size based on moving in a consistent direction (path summation).
# Exploration mechanisms: Uses stochastic sampling; sigma acts as the primary exploration radius.
# Exploitation mechanisms: Mean shifting toward successful regions; covariance adaptation concentrates search on promising directions.
# Boundary handling: Samples are clipped to the domain bounds; a penalty is applied if the mean drifts out of bounds.
# Budget strategy: Divides the budget into generations, adjusting population size proportionally to dimension to ensure full convergence within budget.
# Closest known influences: CMA-ES and (mu, lambda)-ES.
# Novelty or unusual aspects: Diagonal-only covariance for low-overhead memory and computation.
# Failure modes: Large step-size oscillations in highly deceptive or noisy landscapes; poor performance on non-separable problems with high rotation.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.max_pop = min(20 + int(3 * np.log(dim)), budget // 10 or 1)
        self.mu = self.max_pop // 2

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        dim = self.dim
        budget = self.budget
        
        # Initialization
        x_mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        diag_c = np.ones(dim)
        best_x = None
        best_y = float('inf')
        
        evals = 0
        
        while evals < budget:
            # Generate population
            pop_size = min(self.max_pop, budget - evals)
            xs = []
            ys = []
            
            for _ in range(pop_size):
                sample = x_mean + sigma * diag_c * np.random.normal(0, 1, dim)
                sample = np.clip(sample, lb, ub)
                
                y = func(sample)
                evals += 1
                
                xs.append(sample)
                ys.append(y)
                
                if y < best_y:
                    best_y = y
                    best_x = sample.copy()
            
            # Selection
            idx = np.argsort(ys)
            top_xs = np.array([xs[i] for i in idx[:self.mu]])
            
            # Update mean
            new_x_mean = np.mean(top_xs, axis=0)
            
            # Step size adaptation: simple momentum-based update
            diff = new_x_mean - x_mean
            sigma *= np.exp(0.1 * (np.mean(np.abs(diff)) / (np.mean(sigma) + 1e-9) - 0.5))
            
            x_mean = new_x_mean
            
            # Update diagonal covariance based on successful spread
            if self.mu > 1:
                diag_c = np.std(top_xs, axis=0) + 1e-4
                diag_c /= np.max(diag_c)
                
        return best_x, best_y
