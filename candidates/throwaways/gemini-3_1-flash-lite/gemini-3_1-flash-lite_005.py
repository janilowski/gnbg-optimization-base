# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free black-box minimizer using a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) variant with a simplified diagonal covariance structure.
# Search state: Maintains a current mean vector and a diagonal covariance vector (step size control).
# Candidate generation: Samples new points from a Multivariate Normal distribution centered at the mean.
# Selection and replacement: Uses a top-k elitist selection strategy (portion of population) to update the mean through a weighted average.
# Adaptation: Updates the mean toward the best-performing samples; incrementally adjusts the step size based on empirical success rates.
# Exploration mechanisms: Large initial step size (sigma) allows broad search space coverage; stochastic sampling ensures diversity.
# Exploitation mechanisms: Mean shifts toward promising regions; convergence is enforced by shrinking the step size over time.
# Boundary handling: Simple projection (clipping) of candidate points onto the box constraints.
# Budget strategy: Iteratively evaluates population chunks until the function evaluation budget is exhausted.
# Closest known influences: CMA-ES and (1+1)-ES evolution strategies.
# Novelty or unusual aspects: Simplified diagonal adaptation for high robustness across varying dimensions without complex matrix inversion.
# Failure modes: Can get stuck in local optima if the initial step size is too small; ineffective in extremely rugged or discontinuous landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 + int(3 * np.log(dim))
        
    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        else:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
            
        # Initial search parameters
        x = np.random.uniform(lb, ub, self.dim)
        sigma = 0.3 * (ub - lb)
        best_x = np.copy(x)
        best_y = func(best_x)
        evals = 1
        
        # Main optimization loop
        while evals + self.pop_size <= self.budget:
            # Generate population
            candidates = []
            values = []
            
            for _ in range(self.pop_size):
                # Sample and project to bounds
                sample = np.clip(x + np.random.normal(0, sigma, self.dim), lb, ub)
                y = func(sample)
                evals += 1
                
                candidates.append(sample)
                values.append(y)
                
                # Global best tracker
                if y < best_y:
                    best_y = y
                    best_x = np.copy(sample)
            
            # Selection: Sort by performance
            indices = np.argsort(values)
            n_select = self.pop_size // 2
            elite = np.array([candidates[i] for i in indices[:n_select]])
            
            # Adaptation: Move mean toward elite and update step size
            new_x = np.mean(elite, axis=0)
            
            # Simple rank-based step size adaptation
            if values[indices[0]] < best_y:
                sigma *= 1.1  # Success, expand search
            else:
                sigma *= 0.95 # Failure, contract search
                
            x = new_x
            
            # Terminate if convergence is high
            if np.all(sigma < 1e-10):
                break
                
        return best_x, best_y
