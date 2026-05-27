# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (Rastrigin-friendly) random search using an adaptive Gaussian mutation strategy.
# Search state: Maintains a single 'best_x' and 'best_y' found so far, along with an evolving 'step_size' (sigma).
# Candidate generation: Generates new candidates by adding Gaussian noise to the current best_x, scaled by sigma.
# Selection and replacement: Simple elitist replacement; if a candidate is better than current best, it becomes the new best_x.
# Adaptation: The 'sigma' adapts using a 1/5 success rule: it increases if the success rate is high and decreases if low.
# Exploration mechanisms: Driven by the initial high sigma and the random noise component.
# Exploitation mechanisms: Local refinement around the current best_x via sigma reduction.
# Boundary handling: Candidates are clipped to the provided search space bounds.
# Budget strategy: Strict iteration counter; terminates precisely when the budget is reached.
# Closest known influences: (1+1)-Evolution Strategy.
# Novelty or unusual aspects: Minimalist implementation of adaptive step size for robustness in higher dimensions.
# Failure modes: Can get trapped in deep local minima if the basin of attraction is narrow and the initial sigma is poorly scaled.
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
        
        # Adaptive parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        # Main optimization loop
        for i in range(1, self.budget):
            # Generate candidate
            candidate = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
            candidate_y = func(candidate)
            
            # Selection
            if candidate_y < best_y:
                best_x = candidate
                best_y = candidate_y
                success_count += 1
            
            # Adapt sigma every 10 iterations (1/5 success rule approximation)
            if i % 10 == 0:
                if success_count / 10 > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                success_count = 0
                
        return best_x, best_y
