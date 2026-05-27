# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple, robust (1+1)-Evolution Strategy with adaptive step-size control.
# Search state: Stores the best solution found so far and the current mutation step size.
# Candidate generation: Generates a new candidate by adding Gaussian noise to the current best solution.
# Selection and replacement: Uses a greedy selection mechanism: if the new candidate is better, it replaces the current best.
# Adaptation: Employs the 1/5th success rule to adjust the step size: increases after successful steps, decreases after failures.
# Exploration mechanisms: Controlled by the current step size which decays or grows based on progress.
# Exploitation mechanisms: Local search around the current best candidate.
# Boundary handling: Clamps candidates back to the feasible region defined by the function bounds.
# Budget strategy: Exhausts the total budget by performing one iteration per function call until the budget is depleted.
# Closest known influences: (1+1)-ES with 1/5th success rule.
# Novelty or unusual aspects: Minimalist implementation designed for robustness across various black-box landscapes.
# Failure modes: Can get trapped in local optima in highly multi-modal landscapes or suffer from slow convergence in large dimensions.
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

        # Initialization
        x = np.random.uniform(lb, ub)
        y = func(x)
        self.budget -= 1
        
        # Best state tracking
        best_x, best_y = np.copy(x), y
        
        # Step size control
        sigma = 0.2 * (ub - lb)
        success_count = 0
        gen_count = 0
        
        while self.budget > 0:
            # Generate candidate
            candidate = x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            f_val = func(candidate)
            self.budget -= 1
            
            # Selection
            if f_val < y:
                x, y = candidate, f_val
                success_count += 1
                if f_val < best_y:
                    best_x, best_y = np.copy(candidate), f_val
            
            # 1/5th success rule adaptation
            gen_count += 1
            if gen_count >= 5:
                if success_count / gen_count > 0.2:
                    sigma *= 1.2
                elif success_count / gen_count < 0.2:
                    sigma *= 0.8
                success_count = 0
                gen_count = 0
                
        return best_x, best_y
