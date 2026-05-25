# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (1+1)-Evolution Strategy with adaptive step-size control.
# Search state: Stores the current best solution (x) and the current step size (sigma).
# Candidate generation: Generates a new candidate by adding Gaussian noise scaled by sigma to the current best.
# Selection and replacement: Deterministically accepts the candidate if it improves the objective value.
# Adaptation: Employs an 1/5-th success rule: increases sigma if success rate is high, decreases if low.
# Exploration mechanisms: Controlled by sigma; global search initially, local refinement as sigma shrinks.
# Exploitation mechanisms: Local hill-climbing via successful mutations.
# Boundary handling: Candidates are clipped to the box constraints to remain valid.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: (1+1)-ES, CMA-ES (simplified).
# Novelty or unusual aspects: Extremely compact implementation for black-box environments.
# Failure modes: Can get stuck in local optima; ineffective for highly discontinuous or deceptive landscapes.
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
        
        # Initial sigma as fraction of domain width
        sigma = 0.2 * (ub - lb)
        
        best_x, best_y = np.copy(x), y
        evals = 1
        
        # Track success for 1/5-th rule adaptation
        success_count = 0
        window = 10
        
        while evals < self.budget:
            # Generate candidate
            mutation = np.random.normal(0, sigma, self.dim)
            candidate = np.clip(x + mutation, lb, ub)
            
            # Evaluate objective
            y_cand = func(candidate)
            evals += 1
            
            # Selection
            if y_cand < y:
                x, y = candidate, y_cand
                success_count += 1
                if y < best_y:
                    best_x, best_y = np.copy(x), y
            
            # Adapt sigma periodically
            if evals % window == 0:
                if success_count / window > 0.2:
                    sigma *= 1.2  # Too successful, expand search
                elif success_count / window < 0.2:
                    sigma *= 0.8  # Too unsuccessful, contract search
                success_count = 0
                
        return best_x, best_y
