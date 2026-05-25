# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple yet effective (1+1)-Evolution Strategy with adaptive step-size control.
# Search state: Maintains a single best solution (incumbent) and a current step-size (sigma).
# Candidate generation: Generates a new candidate by adding Gaussian noise to the incumbent scaled by sigma.
# Selection and replacement: Greedy selection; the candidate replaces the incumbent if it yields a lower function value.
# Adaptation: Employs the 1/5th success rule, increasing sigma upon success and decreasing it upon failure.
# Exploration mechanisms: Initial sigma is set to a fraction of the search space, ensuring broad coverage.
# Exploitation mechanisms: Fine-grained local search via the shrinking step-size.
# Boundary handling: Candidates are clipped to the defined lower/upper bounds.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: (1+1)-ES, CMA-ES (simplified).
# Novelty or unusual aspects: Extremely compact implementation ideal for black-box benchmarks.
# Failure modes: Can get stuck in local optima; performance sensitive to initial sigma and success rates.
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
        
        # Initialize
        x = np.random.uniform(lb, ub)
        y = func(x)
        self.budget -= 1
        
        # Adaptive step size (1/5th success rule)
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        while self.budget > 0:
            # Generate mutation
            z = np.random.normal(0, 1, self.dim)
            x_next = np.clip(x + sigma * z, lb, ub)
            
            y_next = func(x_next)
            self.budget -= 1
            
            # Selection
            if y_next <= y:
                x, y = x_next, y_next
                success_count += 1
            else:
                success_count -= 0.25 # Penalty for failure
            
            # Adaptation logic (every few steps or continuous adjustment)
            # Adjust sigma every dim steps to keep adaptive behavior stable
            if self.budget % self.dim == 0:
                if success_count > self.dim / 5:
                    sigma *= 1.2
                elif success_count < self.dim / 5:
                    sigma *= 0.8
                success_count = 0
                
        return x, y
