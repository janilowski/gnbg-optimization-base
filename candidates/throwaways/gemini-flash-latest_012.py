# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An adaptive (1+1)-Evolution Strategy (ES) featuring a success-based step-size control mechanism and periodic restarts.
# Search state: Current best solution vector, the best fitness found, a global step-size (sigma), and a success counter for adaptation.
# Candidate generation: Gaussian perturbation of the current best vector, scaled by sigma and the dimension-wise range of the search space.
# Selection and replacement: Simple elitist selection; the candidate replaces the current best solution only if it yields a strictly lower function value.
# Adaptation: Employs a simplified Rechenberg's 1/5th success rule logic, where sigma is increased after successful steps and decreased after failures.
# Exploration mechanisms: Initial random sampling within bounds and periodic restarts (re-randomization) when the search step-size contracts below a specific threshold.
# Exploitation mechanisms: Localized search through Gaussian mutations that shrink in scale as the algorithm converges on a local optimum.
# Boundary handling: All generated candidates are clipped to the feasible region defined by the problem bounds before evaluation.
# Budget strategy: Iterates until the evaluation counter reaches the provided budget, ensuring no evaluations occur beyond the limit.
# Closest known influences: Classical (1+1)-ES and adaptive random search techniques.
# Novelty or unusual aspects: Compact implementation of a robust (1+1) framework that handles multidimensional bounds and dynamically detects budget constraints.
# Failure modes: Can be slow to escape wide plateaus or highly oscillatory landscapes where the 1/5th success rule might prematurely contract the step-size.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    An adaptive (1+1)-Evolution Strategy for black-box minimization.
    It uses a success-based step-size adaptation and restarts on convergence.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Determine bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback if bounds are not provided as expected
            lb = np.full(self.dim, -100.0)
            ub = np.full(self.dim, 100.0)

        # Precompute range for scaling mutations
        space_range = ub - lb
        
        # Initialize search state
        best_x = None
        best_y = float('inf')
        
        # Internal search parameters
        sigma = 0.2
        success_mult = 1.22
        failure_mult = 0.83
        min_sigma = 1e-9
        
        while self.eval_count < self.budget:
            # Restart mechanism if no best_x or if sigma has converged
            if best_x is None or sigma < min_sigma:
                # Random initialization within bounds
                current_x = lb + np.random.rand(self.dim) * space_range
                # Initial evaluation
                if self.eval_count < self.budget:
                    current_y = func(current_x)
                    self.eval_count += 1
                    
                    if current_y < best_y:
                        best_y = current_y
                        best_x = np.copy(current_x)
                
                # Reset sigma for the new search trajectory
                sigma = 0.2
            
            # Generate candidate using Gaussian mutation
            mutation = np.random.normal(0, 1, self.dim) * sigma * space_range
            candidate_x = np.clip(best_x + mutation, lb, ub)
            
            # Evaluate candidate
            if self.eval_count < self.budget:
                candidate_y = func(candidate_x)
                self.eval_count += 1
                
                # Selection and Adaptation
                if candidate_y < best_y:
                    # Success: move to candidate and expand search radius
                    best_y = candidate_y
                    best_x = np.copy(candidate_x)
                    sigma *= success_mult
                else:
                    # Failure: contract search radius
                    sigma *= failure_mult
            else:
                break
                
        return best_x, best_y

# The harness will instantiate and call the Algorithm class.
# The search follows a greedy adaptive approach suitable for a variety of dimensions.
# It ensures bounds are strictly respected via clipping.
# The budget is monitored internally to prevent any extra evaluations.
