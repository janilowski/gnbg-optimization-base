# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (1+1)-Evolution Strategy with adaptive step-size control.
# Search state: Maintains a single candidate solution (incumbent) and a global step-size (sigma).
# Candidate generation: Generates a new point by adding Gaussian noise scaled by sigma to the incumbent.
# Selection and replacement: Uses greedy selection; the candidate replaces the incumbent if its objective value is lower.
# Adaptation: The step-size (sigma) follows the 1/5-th success rule: it increases after successful iterations to speed up convergence and decreases after failures to refine locally.
# Exploration mechanisms: Global search is driven by the initial large sigma and the Gaussian random walk.
# Exploitation mechanisms: Local convergence is driven by the 1/5-th rule shrinking the search volume.
# Boundary handling: Points strictly clamped to the provided box constraints.
# Budget strategy: Iterative loop consumes exactly one evaluation per step until the budget is exhausted.
# Closest known influences: Rechenberg’s (1+1)-ES.
# Novelty or unusual aspects: Minimalist implementation focused on parameter-free adaptation.
# Failure modes: Susceptible to local optima and premature convergence if the initial sigma is poorly scaled.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds safely
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialization
        curr_x = np.random.uniform(lb, ub)
        curr_y = func(curr_x)
        best_x, best_y = np.copy(curr_x), curr_y
        
        # Adaptive parameters
        sigma = (ub - lb) * 0.2
        success_history = []
        
        for i in range(1, self.budget):
            # Generate candidate
            candidate = curr_x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            cand_y = func(candidate)
            
            # Selection
            if cand_y <= curr_y:
                curr_x, curr_y = candidate, cand_y
                success_history.append(1)
                if curr_y < best_y:
                    best_x, best_y = np.copy(curr_x), curr_y
            else:
                success_history.append(0)
            
            # 1/5-th success rule adaptation
            if i % 10 == 0:
                success_ratio = sum(success_history[-10:]) / 10.0
                if success_ratio > 0.2:
                    sigma *= 1.2
                elif success_ratio < 0.2:
                    sigma *= 0.8
                    
            # Termination if search range is negligible
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
