# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust (1+1)-Evolution Strategy (ES) with 1/5th success rule adaptation and periodic restarts.
# Search state: Current best candidate (x), current objective value (y), step size (sigma), and success tracking counters.
# Candidate generation: Gaussian perturbation centered at the current best point, scaled by the adaptive step size.
# Selection and replacement: Greedy selection; the candidate replaces the current best only if its objective value is strictly lower.
# Adaptation: The step size (sigma) is adjusted every 'k' iterations. If the success rate is > 0.2, sigma increases; if < 0.2, it decreases.
# Exploration mechanisms: Large initial sigma (20% of the domain range) and restarts from new random locations once the step size converges or the local search stagnates.
# Exploitation mechanisms: Local Gaussian search that contracts sigma upon failure to find improvements, allowing for fine-grained convergence.
# Boundary handling: Candidates are clipped to the hyper-rectangle defined by the function's lower and upper bounds.
# Budget strategy: Iterates until the evaluation count reaches the specified budget, ensuring no evaluations are wasted while strictly respecting the limit.
# Closest known influences: Rechenberg's (1+1)-ES, Basic Evolution Strategies.
# Novelty or unusual aspects: Combines the simplicity of (1+1)-ES with a restart wrapper to handle multi-modal landscapes within a constrained budget.
# Failure modes: May converge slowly on extremely high-dimensional ridge functions or highly non-separable landscapes compared to covariance-matrix adapting methods.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the optimization algorithm.
        
        Args:
            budget: Total number of function evaluations allowed.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Executes the optimization process.
        
        Args:
            func: The objective function to minimize.
        
        Returns:
            tuple: (best_x, best_y) found during the search.
        """
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)

        range_width = ub - lb
        evals = 0
        
        # Global best trackers
        overall_best_x = None
        overall_best_y = float('inf')

        # Parameters for (1+1)-ES with adaptation
        # Adjustment interval
        k = max(10, self.dim)
        # Success rule multipliers
        c_increase = 1.5
        c_decrease = 1.5**(-0.25)

        while evals < self.budget:
            # Random initialization for a new restart
            x = lb + np.random.rand(self.dim) * range_width
            y = func(x)
            evals += 1
            
            if y < overall_best_y:
                overall_best_y = y
                overall_best_x = x.copy()
                
            if evals >= self.budget:
                break

            # Initial local step size (20% of range)
            sigma = 0.2 * np.max(range_width)
            
            # Local search loop
            success_count = 0
            steps_since_adjustment = 0
            
            while evals < self.budget:
                # Generate candidate
                noise = np.random.randn(self.dim)
                x_trial = x + sigma * noise
                
                # Boundary handling
                x_trial = np.clip(x_trial, lb, ub)
                
                # Evaluation
                y_trial = func(x_trial)
                evals += 1
                steps_since_adjustment += 1
                
                if y_trial < y:
                    y = y_trial
                    x = x_trial
                    success_count += 1
                    
                    if y < overall_best_y:
                        overall_best_y = y
                        overall_best_x = x.copy()
                
                # Adaptation of sigma using 1/5th success rule
                if steps_since_adjustment >= k:
                    success_rate = success_count / steps_since_adjustment
                    if success_rate > 0.2:
                        sigma *= c_increase
                    else:
                        sigma *= c_decrease
                    
                    # Reset counters for next window
                    success_count = 0
                    steps_since_adjustment = 0
                
                # Termination of local search (restart condition)
                # Restart if sigma is negligible relative to domain or if no success
                if sigma < 1e-9 * np.max(range_width):
                    break

        return overall_best_x, overall_best_y
