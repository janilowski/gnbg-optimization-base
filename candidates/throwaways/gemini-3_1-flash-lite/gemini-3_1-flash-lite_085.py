# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (1+1)-Evolution Strategy with adaptive step-size control.
# Search state: Maintains a single current best candidate solution and a scalar step-size (sigma).
# Candidate generation: Generates a new candidate by mutating the current best with Gaussian noise scaled by sigma.
# Selection and replacement: Deterministic replacement: the new candidate replaces the current if its objective value is lower.
# Adaptation: Employs the 1/5th success rule to adjust sigma: increases it if the success rate is high, decreases it otherwise.
# Exploration mechanisms: Gaussian mutation provides local search; sigma adaptation balances the step size.
# Exploitation mechanisms: Local improvement by keeping better solutions.
# Boundary handling: Candidates are clipped to the provided lower and upper bounds.
# Budget strategy: A simple loop that terminates when the function evaluation count reaches the budget.
# Closest known influences: (1+1)-ES, Rechenberg's 1/5th success rule.
# Novelty or unusual aspects: Minimalist implementation designed for robustness in black-box optimization.
# Failure modes: Can get trapped in local optima; performance is sensitive to the initial sigma estimate.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        evals = 1
        
        # Strategy parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        best_x = np.copy(x)
        best_y = y
        
        while evals < self.budget:
            # Mutation: generate new candidate
            candidate = x + np.random.normal(0, sigma, self.dim)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            cand_y = func(candidate)
            evals += 1
            
            # Selection
            if cand_y <= y:
                x = np.copy(candidate)
                y = cand_y
                success_count += 1
                if y < best_y:
                    best_x = np.copy(x)
                    best_y = y
            
            # Adapt sigma every 10 iterations (1/5th success rule)
            if evals % 10 == 0:
                if success_count / 10 > 0.2:
                    sigma *= 1.2
                elif success_count / 10 < 0.2:
                    sigma /= 0.8
                success_count = 0
                
        return best_x, best_y
