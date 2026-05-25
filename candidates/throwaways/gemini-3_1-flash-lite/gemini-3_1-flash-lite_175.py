# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk (Hill Climbing).
# Search state: Maintains a single 'best_x' solution and a current 'step_size' (sigma).
# Candidate generation: Generates new candidates by adding Gaussian noise to the current best_x, scaled by sigma.
# Selection and replacement: Greedy selection; if a candidate improves the best_y, update best_x and increase sigma.
# Adaptation: Employs an adaptive step-size mechanism: doubles sigma on success, halves it on stagnation.
# Exploration mechanisms: Initial sigma is large (fraction of bounds); noise-based perturbation.
# Exploitation mechanisms: Hill climbing focuses on refining the current best solution by shrinking sigma.
# Boundary handling: Candidates are clipped to the defined search space before evaluation.
# Budget strategy: Iterations continue until the function call budget is exhausted.
# Closest known influences: (1+1)-ES (Evolution Strategy).
# Novelty or unusual aspects: Extremely compact implementation suitable for black-box constraints.
# Failure modes: Susceptible to local optima; struggles with highly multi-modal functions.
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
        
        # Initialize state
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        
        best_x = np.copy(x)
        best_y = y
        
        # Initial sigma (step size) set to 20% of range
        sigma = 0.2 * (ub - lb)
        evals = 1
        
        # Track success for adaptive step sizing
        success_streak = 0
        
        while evals < self.budget:
            # Generate candidate: Gaussian perturbation
            candidate = best_x + np.random.normal(0, sigma)
            
            # Boundary handling: Clipping
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            c_y = func(candidate)
            evals += 1
            
            if c_y < best_y:
                # Accept candidate
                best_x = np.copy(candidate)
                best_y = c_y
                sigma *= 1.2  # Increase search radius
                success_streak += 1
            else:
                # Reject candidate
                sigma *= 0.5  # Refine search radius
                success_streak = 0
                
            # Reset sigma if too small
            if np.all(sigma < 1e-7):
                sigma = 0.1 * (ub - lb)
                
        return best_x, best_y
