# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact implementation of the (1+1)-CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for derivative-free black-box minimization.
# Search state: Maintains current center 'x', step size 'sigma', and an evolved covariance matrix C (via Cholesky factor 'A').
# Candidate generation: Points are sampled from a multivariate normal distribution centered at 'x' with covariance sigma^2 * C.
# Selection and replacement: Uses a (1+1)-evolution strategy: a single candidate is generated; if it is better than the parent, it replaces it.
# Adaptation: Step size is adjusted using the 1/5th success rule. The covariance matrix is updated using Rank-1 updates based on the successful step direction.
# Exploration mechanisms: Step size control and covariance matrix adaptation allow for global exploration in early stages and fine-tuning in later stages.
# Exploitation mechanisms: The covariance matrix aligns the search distribution with the local topology of the objective function.
# Boundary handling: Candidates are clipped to the provided search space bounds.
# Budget strategy: Deterministic loops consume the budget linearly; the algorithm terminates exactly when the budget reaches zero.
# Closest known influences: Igel (1998) / Hansen (2016) (1+1)-CMA-ES.
# Novelty or unusual aspects: Minimalist implementation using Cholesky factorization for memory and computation efficiency.
# Failure modes: May converge to local optima in multi-modal landscapes; sensitive to non-convex constraints.
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

        # Initialize parameters
        x = np.random.uniform(lb, ub)
        best_x = np.copy(x)
        best_y = func(x)
        self.budget -= 1

        sigma = 0.3 * (ub - lb)
        C = np.eye(self.dim)
        A = np.eye(self.dim)
        
        # Strategy parameters
        cc = 4.0 / (self.dim + 4.0)
        cs = 0.3
        damp = 1.0 + self.dim / 2.0
        
        c_mean = np.zeros(self.dim)
        
        while self.budget > 0:
            # Generate candidate
            z = np.random.standard_normal(self.dim)
            x_cand = x + sigma * np.dot(A, z)
            x_cand = np.clip(x_cand, lb, ub)
            
            y_cand = func(x_cand)
            self.budget -= 1
            
            # Selection/Replacement
            success = y_cand < best_y
            if success:
                best_x, best_y = np.copy(x_cand), y_cand
                
            # Update step size (1/5th rule approximation)
            success_prob = 1.0 if success else 0.0
            sigma *= np.exp((success_prob - 0.2) / damp)
            
            # Update covariance evolution path
            if success:
                c_mean = (1 - cs) * c_mean + np.sqrt(cs * (2 - cs)) * z
                # Rank-1 update for A
                A_update = np.outer(c_mean, c_mean)
                # Simple covariance update logic
                C = (1 - cc) * C + cc * A_update
                # Recompute Cholesky factor
                try:
                    A = np.linalg.cholesky(C + 1e-9 * np.eye(self.dim))
                except np.linalg.LinAlgError:
                    C = np.eye(self.dim)
                    A = np.eye(self.dim)
            
            x = x_cand if success else x
            
        return best_x, best_y
