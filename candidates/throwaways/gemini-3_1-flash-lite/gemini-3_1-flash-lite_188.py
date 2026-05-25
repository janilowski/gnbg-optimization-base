# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk (Hill Climbing).
# Search state: Maintains the current best point found so far.
# Candidate generation: Generates new steps using a multivariate normal distribution centered at the current best.
# Selection and replacement: Greedy replacement; updates the best point if the candidate yields a lower objective value.
# Adaptation: The step size (sigma) is adjusted via the 1/5th success rule to balance exploration and exploitation.
# Exploration mechanisms: Large initial sigma values allow the algorithm to traverse the search space effectively.
# Exploitation mechanisms: Reduces sigma as the algorithm converges to refine the local minimum.
# Boundary handling: Projects candidate solutions back into the valid box constraints if they exceed bounds.
# Budget strategy: Iteratively evaluates points until the budget is fully exhausted.
# Closest known influences: (1+1)-ES (Evolution Strategy).
# Novelty or unusual aspects: Minimalist implementation focusing on robustness in high dimensions.
# Failure modes: Susceptible to local minima in highly deceptive non-convex landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        range_width = ub - lb
        
        # Initialization
        curr_x = np.random.uniform(lb, ub)
        curr_y = func(curr_x)
        eval_count = 1
        
        best_x = np.copy(curr_x)
        best_y = curr_y
        
        # Adaptive step size (sigma)
        sigma = 0.2 * range_width
        success_count = 0
        
        while eval_count < self.budget:
            # Generate candidate
            step = np.random.normal(0, sigma, size=self.dim)
            candidate_x = np.clip(curr_x + step, lb, ub)
            
            # Evaluate
            candidate_y = func(candidate_x)
            eval_count += 1
            
            # Selection
            if candidate_y < curr_y:
                curr_x, curr_y = candidate_x, candidate_y
                success_count += 1
                if curr_y < best_y:
                    best_x, best_y = np.copy(curr_x), curr_y
            
            # 1/5th success rule for step size adaptation
            if eval_count % (self.dim * 5) == 0:
                success_rate = success_count / (self.dim * 5)
                if success_rate < 0.2:
                    sigma *= 0.8
                elif success_rate > 0.2:
                    sigma *= 1.2
                success_count = 0
                
        return best_x, best_y
