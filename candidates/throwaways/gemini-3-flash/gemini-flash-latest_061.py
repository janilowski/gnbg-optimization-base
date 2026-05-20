# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Population-based Evolution Strategy (ES) utilizing diagonal covariance adaptation and cumulative step-size control.
# Search state: Current mean vector, step size (sigma), and a diagonal vector representing coordinate-wise scaling.
# Candidate generation: Samples are drawn from a multivariate normal distribution centered at the mean, scaled by sigma and the diagonal covariance components.
# Selection and replacement: (mu, lambda) truncation selection where the top performers (approximately 25% of the population) are used to update the mean.
# Adaptation: Sigma (step-size) is updated via a Cumulative Step Size Adaptation (CSA) mechanism comparing the evolution path length to its expected value. The diagonal covariance (coordinate-wise scaling) is updated based on the variance of the successful candidates.
# Exploration mechanisms: Initialized with a relatively large sigma and maintained by the stochastic nature of the ES sampling.
# Exploitation mechanisms: Rank-based weighted averaging for the mean update and contraction of sigma when the evolution path suggests convergent behavior.
# Boundary handling: Candidates are clipped to the hypercube defined by the problem bounds before evaluation.
# Budget strategy: Constant population size per generation; the loop terminates immediately when the evaluation budget is exhausted.
# Closest known influences: sep-CMA-ES (Separable Covariance Matrix Adaptation Evolution Strategy).
# Novelty or unusual aspects: A compact, O(N) complexity implementation of covariance adaptation that provides many of the benefits of full CMA-ES for separable or high-dimensional problems without the quadratic computational or memory overhead.
# Failure modes: Highly non-separable (rotated) functions may result in slower convergence compared to full-covariance methods, as the algorithm only adapts coordinate-aligned scales.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the provided function object
        if hasattr(func, 'lower') and func.lower is not None:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback if no bounds are detected, though benchmark provides them
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)
        
        span = ub - lb
        
        # Strategy parameters
        pop_size = 4 + int(3 * np.log(self.dim))
        mu = max(1, pop_size // 4)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Evolution state variables
        mean = lb + span * 0.5
        sigma = 0.2 * np.max(span)
        diag = np.ones(self.dim)
        ps = np.zeros(self.dim) # Path for sigma adaptation
        
        # Adaptation constants
        cs = (mueff + 2) / (self.dim + mueff + 3)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + cs
        ccov = 1 / (self.dim + 1.3)**2 # Learning rate for diagonal
        
        best_x = None
        best_y = float('inf')
        evals_done = 0
        
        # Main optimization loop
        while evals_done < self.budget:
            # Determine how many to sample based on remaining budget
            current_pop = min(pop_size, self.budget - evals_done)
            if current_pop <= 0:
                break
                
            # Sample candidates: mean + sigma * diag * N(0, I)
            z = np.random.standard_normal((current_pop, self.dim))
            x = mean + sigma * (diag * z)
            
            # Boundary handling: Clip to feasible space
            x = np.clip(x, lb, ub)
            
            # Evaluation
            y = np.zeros(current_pop)
            for i in range(current_pop):
                y[i] = func(x[i])
                if y[i] < best_y:
                    best_y = y[i]
                    best_x = x[i].copy()
            
            evals_done += current_pop
            if evals_done >= self.budget:
                break
            
            # Selection: Sort by objective value
            indices = np.argsort(y)
            top_indices = indices[:mu]
            
            # Update mean using weighted average of the top mu candidates
            old_mean = mean.copy()
            mean = np.dot(weights, x[top_indices])
            
            # Cumulative Step-size Adaptation (CSA)
            # Compute the weighted average of the z-vectors (the standardized steps)
            z_mean = np.dot(weights, z[top_indices])
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z_mean
            
            # Adjust sigma based on the length of the evolution path
            # Expected length of N(0, I) vector is approx sqrt(dim)
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp((cs / damps) * (ps_norm / np.sqrt(self.dim) - 1))
            
            # Diagonal Covariance Adaptation
            # We update the 'diag' vector, which represents the standard deviation of each axis
            if mu > 0:
                # Calculate the variance of the successful steps in the standardized space
                diff_sq = ((x[top_indices] - old_mean) / sigma)**2
                weighted_var = np.dot(weights, diff_sq)
                # Update diagonal scaling with a learning rate
                diag = np.sqrt((1 - ccov) * (diag**2) + ccov * weighted_var)
            
            # Numerical stability: prevents step size from collapsing or exploding
            sigma = np.clip(sigma, 1e-14, np.max(span) * 2)
            diag = np.clip(diag, 1e-7, 1e7)

        return best_x, best_y
