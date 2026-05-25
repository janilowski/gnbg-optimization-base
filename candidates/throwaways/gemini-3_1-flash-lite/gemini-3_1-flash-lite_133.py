# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Gaussian adaptation-based (CMA-ES style) black-box minimizer using a simplified rank-based strategy.
# Search state: Maintains a current mean vector and a global step size (sigma).
# Candidate generation: Generates a population of candidate points by sampling from a multivariate Gaussian distribution centered at the mean.
# Selection and replacement: Evaluates the population, selects the best samples, and updates the mean vector toward the weighted average of the best samples.
# Adaptation: Employs a simple step-size heuristic based on success rates to transition between local exploitation and global exploration.
# Exploration mechanisms: Sampling from a distribution with a non-zero sigma ensures stochastic exploration of the search volume.
# Exploitation mechanisms: The mean consistently shifts toward the best observed regions.
# Boundary handling: Points are clipped to the allowed box constraints before evaluation.
# Budget strategy: Precisely tracks function calls and halts immediately when the budget is reached.
# Closest known influences: Evolutionary Strategies (ES) and (1+1)-CMA-ES simplified for compact implementation.
# Novelty or unusual aspects: Minimalist implementation of adaptive sampling without full covariance matrix estimation to keep the footprint small.
# Failure modes: May struggle with highly deceptive fitness landscapes or extremely high-dimensional spaces where local traps are prevalent.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        
        best_x = None
        best_y = float('inf')
        evals = 0
        
        while evals < self.budget:
            # Generate population
            pop = []
            for _ in range(self.pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and project to bounds
                x = np.clip(mean + np.random.normal(0, sigma, self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                pop.append((x, y))
                
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            
            # Sort population by fitness
            pop.sort(key=lambda item: item[1])
            
            # Update mean using the best half of the population (weighted)
            keep = max(1, len(pop) // 2)
            candidates = np.array([p[0] for p in pop[:keep]])
            new_mean = np.mean(candidates, axis=0)
            
            # Update sigma (simple heuristic: if improvement, expand; else, contract)
            if pop[0][1] < best_y:
                sigma *= 1.1
            else:
                sigma *= 0.95
                
            mean = new_mean
            
        return best_x, best_y
