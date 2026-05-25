# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant utilizing a simplified rank-one update for global optimization.
# Search state: Maintains a mean vector, a global step size (sigma), and a simplified identity-based covariance scale.
# Candidate generation: Multivariate normal distribution sampling centered at the current mean.
# Selection and replacement: Fitness-based selection of a subset of elite individuals (truncation selection).
# Adaptation: Updates mean based on the weighted average of elite individuals; adapts step size (1/5th success rule).
# Exploration mechanisms: Gaussian mutation with adaptive step size.
# Exploitation mechanisms: Mean displacement towards high-performing regions.
# Boundary handling: Clamping parameters to specified bounds during evaluation.
# Budget strategy: Iterative generation until the function evaluation budget is exhausted.
# Closest known influences: (1+1)-CMA-ES and basic Evolution Strategies.
# Novelty or unusual aspects: Lightweight implementation without full covariance matrix inversion.
# Failure modes: Premature convergence on narrow peaks or high-dimensional landscapes with ill-conditioned curvatures.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        mean = np.random.uniform(lb, ub, self.dim)
        sigma = 0.2 * (ub - lb)
        pop_size = 4 + int(3 * np.log(self.dim))
        
        best_x = None
        best_y = float('inf')

        def evaluate(x):
            nonlocal best_x, best_y, self
            x_clamped = np.clip(x, lb, ub)
            y = func(x_clamped)
            self.evals += 1
            if y < best_y:
                best_y = y
                best_x = x_clamped
            return y

        # Initial evaluation
        evaluate(mean)

        while self.evals + pop_size <= self.budget:
            # Generate population
            pop = [mean + sigma * np.random.randn(self.dim) for _ in range(pop_size)]
            fitness = [evaluate(x) for x in pop]
            
            # Sort by fitness and select elites
            sorted_indices = np.argsort(fitness)
            elite_indices = sorted_indices[:max(1, pop_size // 2)]
            
            # Update mean (weighted towards better half)
            old_mean = mean.copy()
            mean = np.mean([pop[i] for i in elite_indices], axis=0)
            
            # Adaptive step size (1/5th success rule heuristic)
            if fitness[sorted_indices[0]] < fitness[sorted_indices[-1]]:
                sigma *= 1.05
            else:
                sigma *= 0.95
            
            # Check for convergence
            if np.linalg.norm(mean - old_mean) < 1e-9:
                sigma *= 2.0

        return best_x, best_y
