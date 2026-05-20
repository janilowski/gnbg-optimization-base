# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust, adaptive (1+1)-Evolution Strategy (ES) featuring restarts and dynamic step-size control.
# Search state: Current best solution (x_best, y_best) and the mutation strength (sigma).
# Candidate generation: Generates a single candidate per iteration by applying Gaussian noise scaled by sigma to the current best solution.
# Selection and replacement: Greedy selection; the candidate replaces the current best solution only if it achieves a lower objective value.
# Adaptation: Employs a simplified Rechenberg 1/5th success rule. Sigma increases on success to accelerate convergence and decreases on failure to refine the local search.
# Exploration mechanisms: Occasional restarts with random initialization and reset sigma whenever the step size collapses below a relative threshold.
# Exploitation mechanisms: Gaussian local search with step-size adaptation allows for fine-tuning near local minima.
# Boundary handling: Candidates are clipped to the hypercube defined by the problem bounds.
# Budget strategy: Continues searching until the evaluation counter reaches the provided budget, ensuring no wasted evaluations.
# Closest known influences: Classical (1+1)-ES, Rechenberg's adaptation rule, and multi-start local search.
# Novelty or unusual aspects: Combines extreme simplicity with a restart mechanism to handle multi-modal landscapes without complex population dynamics.
# Failure modes: May converge slowly on highly non-separable or extremely rugged landscapes compared to covariance-matrix adaptation methods.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    An adaptive (1+1)-Evolution Strategy for black-box minimization.
    Designed for robustness across various dimensions and budgets.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Extract bounds from the provided function object
        if hasattr(func, 'bounds') and func.bounds is not None:
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        elif hasattr(func, 'lower') and func.lower is not None:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        else:
            # Fallback to defaults if no bounds are detected
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        range_width = ub - lb
        
        # Internal state variables
        best_x = None
        best_y = float('inf')
        
        # Search parameters
        sigma_init = 0.2 * np.max(range_width)
        sigma_min = 1e-9 * np.min(range_width)
        
        # Multi-start loop
        while self.eval_count < self.budget:
            # Initialize search point and step size for this restart
            curr_x = lb + np.random.rand(self.dim) * range_width
            
            # Initial evaluation for the restart
            if self.eval_count < self.budget:
                curr_y = func(curr_x)
                self.eval_count += 1
                if curr_y < best_y:
                    best_y = curr_y
                    best_x = np.copy(curr_x)
            else:
                break
                
            sigma = sigma_init
            
            # Adaptive 1+1 ES loop
            stagnation_counter = 0
            while self.eval_count < self.budget:
                # Generate candidate
                noise = np.random.standard_normal(self.dim)
                candidate_x = curr_x + sigma * noise
                
                # Boundary handling: Clip to bounds
                candidate_x = np.clip(candidate_x, lb, ub)
                
                # Evaluation
                candidate_y = func(candidate_x)
                self.eval_count += 1
                
                # Update best global and current point
                if candidate_y < curr_y:
                    # Success: Increase sigma
                    sigma *= 1.1
                    curr_x = candidate_x
                    curr_y = candidate_y
                    stagnation_counter = 0
                    
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = np.copy(curr_x)
                else:
                    # Failure: Decrease sigma
                    sigma *= 0.85
                    stagnation_counter += 1
                
                # Restart conditions
                # 1. Sigma becomes too small relative to the search space
                # 2. Too many consecutive failures (stagnation)
                if sigma < sigma_min or stagnation_counter > (100 + 20 * self.dim):
                    break
        
        # Ensure we return the best found even if budget is tight
        return best_x, best_y

# The module is intended for use within a harness that provides the objective function.
# No top-level execution code is included.
