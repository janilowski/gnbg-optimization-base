# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant with a simplified diagonal covariance structure.
# Search state: Maintains a distribution mean and an adaptive step size (sigma) to explore the search space.
# Candidate generation: Samples new points from a multivariate normal distribution centered at the current mean.
# Selection and replacement: Uses a rank-based selection where the top half of the population updates the distribution mean.
# Adaptation: Employs a simple cumulative step-size adaptation (CSA) to adjust global diffusion.
# Exploration mechanisms: Gaussian sampling ensures global and local search capability proportional to sigma.
# Exploitation mechanisms: Mean shifting moves the distribution toward regions of higher fitness found in the sample.
# Boundary handling: Uses simple clipping to the function bounds.
# Budget strategy: Divides the budget into generations, stopping when the last evaluation is consumed.
# Closest known influences: CMA-ES and (1+1)-ES algorithms.
# Novelty or unusual aspects: Diagonal-only covariance structure for memory efficiency and robustness in higher dimensions.
# Failure modes: Can get trapped in local minima in extremely rugged landscapes; sensitive to initialization if the landscape is highly non-convex.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        self.max_gens = max(1, budget // (self.pop_size + 1))

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize strategy parameters
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        best_x = np.copy(mean)
        best_y = float('inf')
        
        evals = 0
        gen = 0
        
        # Evaluation helper
        def evaluate(x):
            nonlocal best_x, best_y, evals
            if evals >= self.budget:
                return best_y
            x = np.clip(x, lb, ub)
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.copy(x)
            return y

        # Initial evaluation
        best_y = evaluate(mean)
        
        # Main optimization loop
        while evals < self.budget and gen < self.max_gens:
            # Sample population
            pop = []
            for _ in range(self.pop_size):
                sample = np.clip(mean + sigma * np.random.randn(self.dim), lb, ub)
                pop.append((sample, evaluate(sample)))
            
            # Sort by fitness (simple elitism / rank)
            pop.sort(key=lambda x: x[1])
            
            # Update mean using a weighted average of the top half
            top_half = pop[:self.pop_size // 2]
            new_mean = np.mean([x[0] for x in top_half], axis=0)
            
            # Update step size based on success
            if pop[0][1] < best_y:
                sigma *= 1.1  # Success, expand search
            else:
                sigma *= 0.8  # Failure, shrink search
                
            mean = new_mean
            gen += 1
            
        return best_x, best_y
