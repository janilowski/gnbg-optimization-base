# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1) Evolution Strategy (ES) featuring an adaptive step size and periodic restarts. 
# Search state: Current best position (x), current best value (y), current mutation strength (sigma), and success history.
# Candidate generation: Perturbs the current best candidate with Gaussian noise scaled by sigma, then clips to bounds.
# Selection and replacement: Simple greedy selection; the candidate replaces the current best if its objective value is lower.
# Adaptation: Employs an adaptation of the 1/5th success rule. Sigma increases on success and decreases on failure to maintain an optimal mutation rate.
# Exploration mechanisms: High initial sigma values and random restarts triggered when the local search converges (sigma becomes negligible).
# Exploitation mechanisms: Gaussian mutation allows for fine-grained local refinement as sigma decreases.
# Boundary handling: Candidates are clipped to the hypercube defined by the problem's lower and upper bounds.
# Budget strategy: Monitored through a counter to ensure the search terminates exactly when the evaluation limit is reached.
# Closest known influences: Rechenberg's (1+1)-ES, specifically the adaptive step-size control.
# Novelty or unusual aspects: Simplified restart logic integrated with the ES to ensure the budget is fully utilized across different regions of the search space.
# Failure modes: May struggle with highly non-separable functions in very high dimensions where the lack of a full covariance matrix prevents efficient diagonal movement.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the (1+1)-ES optimizer.
        :param budget: Total number of function evaluations allowed.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        """
        Executes the optimization process.
        :param func: The objective function to minimize.
        :return: (best_x, best_y)
        """
        # Extract bounds from the function object
        if hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        else:
            lb = np.array(func.lower)
            ub = np.array(func.upper)

        best_x = None
        best_y = float('inf')

        # Constants for step-size adaptation (1/5th success rule derivative)
        increase_factor = 1.5
        decrease_factor = 0.5 ** 0.25  # ~0.84
        
        while self.eval_count < self.budget:
            # Restart / Initialization logic
            curr_x = np.random.uniform(lb, ub, self.dim)
            curr_y = func(curr_x)
            self.eval_count += 1
            
            if curr_y < best_y:
                best_y = curr_y
                best_x = np.copy(curr_x)
                
            if self.eval_count >= self.budget:
                break

            # Initial sigma proportional to the search range
            sigma = 0.2 * np.mean(ub - lb)
            
            # Local search loop
            stagnation_limit = 100 + self.dim * 10
            stagnant_iters = 0
            
            while self.eval_count < self.budget:
                # Generate candidate
                noise = np.random.randn(self.dim) * sigma
                candidate_x = np.clip(curr_x + noise, lb, ub)
                
                # Evaluate
                candidate_y = func(candidate_x)
                self.eval_count += 1
                
                if candidate_y < curr_y:
                    # Success
                    curr_x = candidate_x
                    curr_y = candidate_y
                    sigma *= increase_factor
                    stagnant_iters = 0
                    
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = np.copy(curr_x)
                else:
                    # Failure
                    sigma *= decrease_factor
                    stagnant_iters += 1
                
                # Check for convergence or stagnation to trigger a restart
                if sigma < 1e-12 or stagnant_iters > stagnation_limit:
                    break
        
        return best_x, best_y
