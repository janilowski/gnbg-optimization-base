# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1)-CMA-ES (Evolution Strategy) variant featuring rank-one covariance matrix adaptation and a modified 1/5th success rule for step-size control. 
# Search state: Current best solution vector, a global step size (sigma), a covariance matrix representing the search distribution shape, and success statistics.
# Candidate generation: A single candidate is generated per iteration by sampling from a multivariate normal distribution centered at the current best point, scaled by sigma and the covariance matrix.
# Selection and replacement: A greedy (1+1) selection strategy where the candidate replaces the current best if its objective value is lower or equal.
# Adaptation: The covariance matrix is updated using a rank-one update based on the successful step. The step size (sigma) is adapted based on the success rate over a sliding window to maintain an optimal search pressure.
# Exploration mechanisms: Initialized with a relatively large step size and identity covariance. Global exploration is primarily driven by the initial sampling and the covariance expansion in successful directions.
# Exploitation mechanisms: Covariance adaptation allows the search to align with narrow valleys or ridges. Step size contraction enables precise convergence to local minima.
# Boundary handling: Candidates are clipped to the hypercube defined by the problem bounds before evaluation.
# Budget strategy: The algorithm runs until the evaluation budget is exhausted. If the search converges (sigma becomes effectively zero), the search is restarted from a new random location.
# Closest known influences: The (1+1)-CMA-ES by Igel et al., and the classic 1/5th success rule from Rechenberg.
# Novelty or unusual aspects: Combines rank-one covariance updates with a simplified success-based adaptation for robustness in a compact footprint suitable for black-box benchmarks.
# Failure modes: May struggle with highly multimodal landscapes if the budget is too small to permit sufficient restarts or if the global optimum has a very small basin of attraction.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0
        
        # Hyperparameters for (1+1)-CMA-ES
        self.p_target = 0.2
        self.c_p = 1.0 / (self.dim + 2.0)
        self.c_cov = 2.0 / (self.dim**2 + 6.0)
        self.damp = 1.0 + self.dim / 2.0

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        best_x = None
        best_y = float('inf')

        while self.eval_count < self.budget:
            # Restart/Initialize logic
            x = np.random.uniform(lb, ub)
            y = func(x)
            self.eval_count += 1
            
            if y < best_y:
                best_x, best_y = np.copy(x), y
            
            if self.eval_count >= self.budget:
                break

            # Search parameters
            sigma = 0.2 * np.max(ub - lb)
            pc = np.zeros(self.dim)
            C = np.eye(self.dim)
            ps_success = self.p_target

            # Local search loop
            stagnation_counter = 0
            while self.eval_count < self.budget:
                # Generate candidate
                # We use the decomposition of C or just multivariate_normal
                # Since C is symmetric positive definite:
                try:
                    z = np.random.multivariate_normal(np.zeros(self.dim), C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular
                    C = np.eye(self.dim)
                    z = np.random.normal(0, 1, self.dim)
                
                x_cand = x + sigma * z
                
                # Boundary handling: Clipping
                x_cand = np.clip(x_cand, lb, ub)
                
                y_cand = func(x_cand)
                self.eval_count += 1
                
                # Success check
                success = y_cand <= y
                
                # Update success rate
                ps_success = (1.0 - self.c_p) * ps_success + self.c_p * float(success)
                
                # Update step size sigma
                sigma = sigma * np.exp((ps_success - self.p_target) / (self.damp * (1.0 - self.p_target)))
                
                if success:
                    # Update mean
                    x, y = x_cand, y_cand
                    if y < best_y:
                        best_x, best_y = np.copy(x), y
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                    
                    # Update covariance matrix (rank-one update)
                    # Implementation of (1+1)-CMA-ES covariance update
                    if ps_success < 0.44:
                        pc = (1.0 - self.c_p) * pc + np.sqrt(self.c_p * (2.0 - self.c_p)) * z
                        C = (1.0 - self.c_cov) * C + self.c_cov * np.outer(pc, pc)
                else:
                    stagnation_counter += 1
                
                # Termination conditions for restart
                if sigma < 1e-12 * np.max(ub - lb) or stagnation_counter > 100 + 10 * self.dim:
                    break

        return best_x, best_y
