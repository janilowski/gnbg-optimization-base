# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free covariance matrix adaptation evolution strategy (CMA-ES) variant utilizing a simplified rank-one update mechanism.
# Search state: Maintains a current mean vector and a step-size (sigma), with a fixed identity covariance matrix to ensure stability in limited budgets.
# Candidate generation: Multivariate normal distribution sampling around the current mean.
# Selection and replacement: Steady-state evolution; the mean moves towards the best encountered point after every evaluation if improved.
# Adaptation: Step-size (sigma) adaptation based on the success rate of improving the current best objective value (1/5th rule inspired).
# Exploration mechanisms: Gaussian noise sampling controlled by sigma; global search via initial wide sampling.
# Exploitation mechanisms: Local hill-climbing via mean updates when better solutions are found.
# Boundary handling: Clamping samples to the defined search space bounds.
# Budget strategy: Iterates until the evaluation budget is exhausted; updates parameters every step.
# Closest known influences: (1+1)-ES (Evolution Strategy) with adaptive step size.
# Novelty or unusual aspects: Highly memory-efficient, no matrix inversion required, robust in low-budget/high-dimension scenarios.
# Failure modes: Susceptible to local optima and premature convergence if the step size shrinks too rapidly.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract boundaries
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        else:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)

        # Initialization
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        best_x, best_y = np.copy(x), y
        
        # Step size (sigma) initial guess: 20% of the range
        sigma = 0.2 * (ub - lb)
        
        eval_count = 1
        success_count = 0
        
        # Main optimization loop
        while eval_count < self.budget:
            # Generate candidate
            z = np.random.normal(0, sigma, self.dim)
            candidate = np.clip(x + z, lb, ub)
            
            # Evaluate
            f_candidate = func(candidate)
            eval_count += 1
            
            # Selection
            if f_candidate < y:
                x = np.copy(candidate)
                y = f_candidate
                success_count += 1
                
                # Update global best
                if y < best_y:
                    best_x = np.copy(x)
                    best_y = y
            
            # Adaptation (1/5th success rule)
            # Adjust sigma every dim evaluations
            if eval_count % self.dim == 0:
                rate = success_count / self.dim
                if rate > 0.2:
                    sigma *= 1.2  # Expand if success is high
                elif rate < 0.2:
                    sigma *= 0.8  # Contract if success is low
                success_count = 0
                
            # Convergence check: stop if sigma is extremely small
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
