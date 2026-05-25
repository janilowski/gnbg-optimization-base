# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES-inspired local search algorithm using rank-based weighted adaptation.
# Search state: Maintains a mean vector, a step size (sigma), and an identity covariance matrix (C).
# Candidate generation: Generates candidates via multivariate normal sampling centered at the mean.
# Selection and replacement: Uses (mu, lambda) selection, updating the mean based on the top-performing fraction.
# Adaptation: Updates step size using simple cumulative trend detection and diagonal covariance adaptation.
# Exploration mechanisms: Controlled by sigma and the diagonal covariance matrix to allow axis-aligned scaling.
# Exploitation mechanisms: Mean shifting towards the best candidates discovered in each generation.
# Boundary handling: Clamping proposed candidates back into the provided hyper-rectangle bounds.
# Budget strategy: Iterative generation until the budget is exactly exhausted.
# Closest known influences: CMA-ES and simple Evolutionary Strategies.
# Novelty or unusual aspects: Minimalist implementation without full matrix inversion or complex step-size control.
# Failure modes: Can get trapped in narrow local minima; performance degrades in extremely high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        
        # Initialization
        x_mean = np.random.uniform(lb, ub)
        sigma = 0.2 * (ub - lb)
        diag_d = np.ones(self.dim)
        
        best_x = None
        best_y = float('inf')
        evals = 0
        
        while evals < self.budget:
            # Generate candidates
            population = []
            f_values = []
            
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Sample
                z = np.random.normal(0, 1, self.dim)
                x = x_mean + sigma * (diag_d * z)
                x = np.clip(x, lb, ub)
                
                y = func(x)
                evals += 1
                
                population.append(x)
                f_values.append(y)
                
                if y < best_y:
                    best_y = y
                    best_x = x
            
            if not f_values: break
            
            # Selection
            idx = np.argsort(f_values)
            top_idx = idx[:mu]
            top_x = np.array([population[i] for i in top_idx])
            
            # Update mean
            old_mean = x_mean.copy()
            x_mean = np.sum(top_x.T * weights, axis=1)
            
            # Simple Adaptation
            # Update step size based on movement
            dist = np.linalg.norm(x_mean - old_mean)
            sigma *= np.exp(0.1 * (dist / (sigma.mean() + 1e-9) - 0.5))
            
            # Update spread along axes based on success
            if mu > 1:
                diff = (top_x - x_mean)
                diag_d = np.sqrt(np.mean(diff**2, axis=0)) + 1e-9
                diag_d /= np.max(diag_d)
                
        return best_x, best_y
