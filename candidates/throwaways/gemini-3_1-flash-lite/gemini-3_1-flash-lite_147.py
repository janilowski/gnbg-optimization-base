# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant with a simplified diagonal covariance estimate.
# Search state: Population mean, step size (sigma), and diagonal covariance matrix components.
# Candidate generation: Multivariate normal distribution sampling centered at the current mean.
# Selection and replacement: Top 30% of population (rank-based selection) updates the distribution parameters.
# Adaptation: Moving average updates for the mean and diagonal covariance; adaptive step-size scaling based on success rate.
# Exploration mechanisms: Large initial sigma and stochastic sampling ensure global coverage.
# Exploitation mechanisms: Mean shifting towards successful candidates and adaptive diagonal scaling.
# Boundary handling: Clipping values to the provided lower and upper bounds during candidate generation.
# Budget strategy: Strict iteration loop limited by the budget parameter.
# Closest known influences: CMA-ES and simple Evolution Strategies.
# Novelty or unusual aspects: Diagonal-only covariance update for memory efficiency and robustness in high dimensions without full matrix inversion.
# Failure modes: Can get stuck in local optima if the function landscape is highly deceptive or if the step size shrinks prematurely.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        pop_size = 4 + int(3 * np.log(self.dim))
        mu = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        diag_c = np.ones(self.dim)
        
        best_x = None
        best_y = float('inf')
        evals = 0

        while evals < self.budget:
            # Generate candidates
            population = []
            for _ in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and clip
                noise = np.random.normal(0, 1, self.dim)
                candidate = mu + sigma * (noise * np.sqrt(diag_c))
                candidate = np.clip(candidate, lb, ub)
                
                y = func(candidate)
                evals += 1
                
                if y < best_y:
                    best_y = y
                    best_x = candidate
                
                population.append((y, candidate))

            # Selection: Sort by fitness
            population.sort(key=lambda x: x[0])
            elite = population[:max(1, pop_size // 3)]
            
            # Update mean
            old_mu = mu.copy()
            mu = np.mean([x[1] for x in elite], axis=0)
            
            # Adaptation: Update diagonal covariance using movement
            diff = (mu - old_mu) / sigma
            diag_c = 0.9 * diag_c + 0.1 * (diff**2 + 0.1)
            
            # Update step size (1/5th rule inspired)
            if population[0][0] < population[pop_size//2][0]:
                sigma *= 1.1
            else:
                sigma *= 0.9
            
            # Decay constraints
            sigma = np.clip(sigma, 1e-7, (ub - lb))

        return best_x, best_y
