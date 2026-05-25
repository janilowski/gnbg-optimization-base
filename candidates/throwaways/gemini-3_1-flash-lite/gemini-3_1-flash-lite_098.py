# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A minimalist implementation of a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant utilizing a simplified diagonal covariance approach.
# Search state: Maintains a Gaussian distribution defined by a mean vector and a step size (sigma).
# Candidate generation: Samples population size proportional to dimension from a multivariate normal distribution.
# Selection and replacement: Selects the best 50% of the population, updating the mean through a weighted average (recombination).
# Adaptation: Updates step size based on success rate (1/5th rule) and distribution mean via moving average.
# Exploration mechanisms: Initialized with a large sigma; controlled stochastic sampling ensures space coverage.
# Exploitation mechanisms: Mean shifts toward the best observed regions; sigma contracts as the distribution narrows.
# Boundary handling: Projections (clipping) are applied to candidates before function evaluation.
# Budget strategy: Divides budget into generations; stops early if budget is exhausted.
# Closest known influences: CMA-ES and (μ, λ)-ES.
# Novelty or unusual aspects: Uses a simplified diagonal step-size adaptation for low-overhead performance.
# Failure modes: May converge to local optima in highly deceptive landscapes or struggle with extreme objective scaling.
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

        # Algorithm parameters
        pop_size = 4 + int(3 * np.log(self.dim))
        sigma = 0.3 * (ub - lb)
        mean = np.random.uniform(lb, ub)
        
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
                
                # Sample and project to bounds
                x = np.clip(mean + np.random.normal(0, sigma, self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                population.append((x, y))
                if y < best_y:
                    best_y = y
                    best_x = x

            # Sort by performance
            population.sort(key=lambda item: item[1])
            
            # Recombination: Update mean using the best half
            keep = max(1, pop_size // 2)
            new_mean = np.mean([p[0] for p in population[:keep]], axis=0)
            
            # Step size adaptation (1/5th rule approximation)
            # If the best point improved, increase sigma, else shrink
            if population[0][1] < best_y:
                sigma *= 1.2
            else:
                sigma *= 0.8
            
            mean = new_mean
            
            # Convergence check: sigma small relative to bounds
            if np.mean(sigma) < 1e-9:
                break
                
        return best_x, best_y
