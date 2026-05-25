# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES-like evolution strategy using an isotropic mutation distribution.
# Search state: Tracks a central mean vector and a step size (sigma) which adapts based on success.
# Candidate generation: Samples population members from a multivariate normal distribution centered at the mean.
# Selection and replacement: Uses a (mu, lambda)-style selection where the best fraction of the population updates the mean.
# Adaptation: Employs a 1/5th success rule to adjust the step size (sigma) to maintain ideal convergence speed.
# Exploration mechanisms: Initialized with a large sigma relative to the domain bounds; controlled by stochastic perturbation.
# Exploitation mechanisms: The mean shifts toward favorable regions, and sigma contracts as the population clusters.
# Boundary handling: Projects candidates back into the feasible domain using clipping (clamping).
# Budget strategy: Uses a fixed population size per generation, iterating until the budget is exhausted.
# Closest known influences: Simplified (1+lambda)-ES and CMA-ES principles.
# Novelty or unusual aspects: Minimalist implementation of adaptive step-size logic without complex matrix inversion.
# Failure modes: Can get stuck in local optima; performance sensitive to initial step-size scaling.
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
        center = np.random.uniform(lb, ub)
        sigma = 0.2 * (ub - lb)
        best_x = np.copy(center)
        best_y = float('inf')
        
        # Strategy parameters
        pop_size = 4 + int(3 * np.log(self.dim))
        evals_remaining = self.budget
        
        while evals_remaining > 0:
            # Generate candidates
            population = []
            for _ in range(min(pop_size, evals_remaining)):
                candidate = center + np.random.normal(0, sigma, self.dim)
                # Boundary clamping
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                
                if val < best_y:
                    best_y = val
                    best_x = np.copy(candidate)
                
                population.append((candidate, val))
                evals_remaining -= 1
            
            # Selection: Sort by value
            population.sort(key=lambda x: x[1])
            
            # Update mean (Selection)
            elite_count = max(1, pop_size // 2)
            new_mean = np.mean([p[0] for p in population[:elite_count]], axis=0)
            
            # Adaptation: 1/5th success rule
            success = population[0][1] < best_y
            if success:
                sigma *= 1.2
            else:
                sigma *= 0.8
            
            center = new_mean
            
            # Early exit if sigma is negligible
            if np.mean(sigma) < 1e-12:
                break
                
        return best_x, best_y
