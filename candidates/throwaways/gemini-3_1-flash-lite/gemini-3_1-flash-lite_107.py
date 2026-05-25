# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free CMA-ES-inspired local search algorithm using rank-based selection and adaptive step size control.
# Search state: Maintains a current mean vector and a global step size (sigma).
# Candidate generation: Samples individuals from a multivariate normal distribution centered at the mean with covariance defined by sigma.
# Selection and replacement: Evaluates population, sorts by fitness, and uses the best weighted individuals to update the mean.
# Adaptation: Updates sigma via the $1/5$-th success rule (increasing if >20% of samples improve, decreasing otherwise).
# Exploration mechanisms: Stochastic sampling enables escaping local minima via breadth.
# Exploitation mechanisms: Mean shift toward the best samples allows local convergence.
# Boundary handling: Clamps generated samples to the provided domain bounds.
# Budget strategy: Exhaustive usage; divides budget into population-based generations until exhaustion.
# Closest known influences: Simplified CMA-ES and (mu, lambda)-ES.
# Novelty or unusual aspects: Extremely compact implementation using standard library and numpy.
# Failure modes: May converge prematurely in highly non-convex or needle-in-a-haystack landscapes.
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
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = None
        best_y = float('inf')
        
        evals = 0
        
        while evals < self.budget:
            # Generate offspring
            samples = []
            for _ in range(pop_size):
                if evals >= self.budget: break
                
                # Sample and clamp
                x = np.clip(mean + sigma * np.random.normal(0, 1, self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                samples.append((x, y))
                
                if y < best_y:
                    best_y = y
                    best_x = x

            # Sort offspring
            samples.sort(key=lambda item: item[1])
            
            # Update mean using a fraction of the population
            # (Weighted average of top 50% of the population)
            num_elite = max(1, pop_size // 2)
            weights = np.array([1.0 / (i + 1) for i in range(num_elite)])
            weights /= weights.sum()
            
            new_mean = np.zeros(self.dim)
            for i in range(num_elite):
                new_mean += weights[i] * samples[i][0]
            
            # Adaptation: 1/5th success rule
            if samples[0][1] < best_y:
                sigma *= 1.2
            else:
                sigma *= 0.8
                
            mean = new_mean
            
            # Restart mechanism if sigma vanishes
            if np.all(sigma < 1e-10):
                sigma = 0.3 * (ub - lb)
                mean = np.random.uniform(lb, ub)

        return best_x, best_y
