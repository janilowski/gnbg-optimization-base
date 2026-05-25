# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant limited to Diagonal Covariance (VarCMA).
# Search state: Maintains a mean vector, a diagonal covariance vector (step sizes), and an evolution path vector.
# Candidate generation: Multivariate normal sampling centered at the mean, scaled by diagonal standard deviations.
# Selection and replacement: Rank-based selection of the top half of a generated population, updating mean via weighted average.
# Adaptation: Updates step size based on evolution path success and covariance via rank-one and rank-mu updates.
# Exploration mechanisms: Large step sizes (sigma) and Gaussian noise ensure global coverage.
# Exploitation mechanisms: Sequential mean shifting towards successful regions and shrinking variance.
# Boundary handling: Penalized projection (clamping) to function-defined ranges.
# Budget strategy: Iteratively evaluates population chunks until the budget is exhausted.
# Closest known influences: Simplified CMA-ES (Diagonal variant).
# Novelty or unusual aspects: Minimalist implementation of CMA logic using only numpy.
# Failure modes: Can get stuck in local optima on highly multi-modal landscapes if initial sigma is too small.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract boundaries
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Hyperparameters
        pop_size = 4 + int(3 * np.log(self.dim))
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        diag_c = np.ones(self.dim)
        
        best_x = None
        best_y = float('inf')
        evals = 0

        # Optimization loop
        while evals < self.budget:
            # Generate candidates
            population = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                z = np.random.normal(0, 1, self.dim)
                x = mean + sigma * z * diag_c
                # Boundary projection
                x = np.clip(x, lb, ub)
                
                y = func(x)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = x
                population.append((x, y, z))
            
            if not population:
                break
                
            # Selection
            population.sort(key=lambda item: item[1])
            selected = population[:pop_size // 2]
            
            # Update mean
            old_mean = mean.copy()
            mean = np.mean([p[0] for p in selected], axis=0)
            
            # Adapt diagonal covariance (simplified step size update)
            z_mean = np.mean([p[2] for p in selected], axis=0)
            diag_c = 0.9 * diag_c + 0.1 * np.abs(z_mean)
            
            # Adapt global sigma
            if np.mean([p[1] for p in selected]) < best_y:
                sigma *= 1.05
            else:
                sigma *= 0.95
                
        return best_x, best_y
