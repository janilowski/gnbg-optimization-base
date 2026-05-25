# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk (similar to a (1+1)-ES).
# Search state: Maintains the 'best' solution found so far and an adaptive step size (sigma).
# Candidate generation: Generates a new candidate by adding Gaussian noise scaled by sigma to the current best solution.
# Selection and replacement: Simple elitist replacement: if the candidate is better than the incumbent, it becomes the new current solution.
# Adaptation: Employs a 1/5th success rule logic to adapt sigma: increase if convergence is fast, decrease if slow.
# Exploration mechanisms: Initial sigma is large (fraction of search space width) to explore globally before refining.
# Exploitation mechanisms: Local hill-climbing via adaptive Gaussian step shrinkage.
# Boundary handling: Candidates are clipped to the provided lower and upper bounds using numpy projection.
# Budget strategy: Stops exactly when the function evaluation count reaches the provided budget.
# Closest known influences: (1+1)-Evolution Strategy with 1/5th success rule.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness and zero dependencies.
# Failure modes: Can get trapped in local optima for highly multi-modal functions.
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
        
        best_x = np.copy(x)
        best_y = y
        
        # Initial sigma as 20% of the range
        sigma = 0.2 * (ub - lb)
        evals = 1
        
        # Adaptation parameters
        success_count = 0
        
        while evals < self.budget:
            # Generate candidate
            step = np.random.normal(0, sigma, self.dim)
            candidate = np.clip(best_x + step, lb, ub)
            
            # Evaluate
            candidate_y = func(candidate)
            evals += 1
            
            # Selection
            if candidate_y < best_y:
                best_x = candidate
                best_y = candidate_y
                success_count += 1
            
            # 1/5th success rule adaptation every 10 steps
            if evals % 10 == 0:
                if success_count / 10 > 0.2:
                    sigma *= 1.1
                elif success_count / 10 < 0.2:
                    sigma *= 0.9
                success_count = 0
            
            if evals >= self.budget:
                break
                
        return best_x, best_y
