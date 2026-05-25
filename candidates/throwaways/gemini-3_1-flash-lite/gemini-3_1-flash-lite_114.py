# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (A)daptive (S)tep size (SAS).
# Search state: Maintains a current best solution and an adaptive step size (sigma).
# Candidate generation: New candidates are sampled from a multivariate normal distribution centered at the current best.
# Selection and replacement: Simple greedy update: if a candidate improves the best objective, it replaces it.
# Adaptation: The step size (sigma) follows a 1/5th success rule: increases when updates are frequent, decreases when rare.
# Exploration mechanisms: Stochastic Gaussian mutations allow broad global exploration.
# Exploitation mechanisms: Shrinking sigma focuses the search around successful local regions.
# Boundary handling: Clipping ensures all candidates remain within the specified search space.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: Evolutionary strategies (ES), (1+1)-ES.
# Novelty or unusual aspects: Minimalist implementation of adaptive step-size control without coordinate transformation.
# Failure modes: Susceptible to getting trapped in local optima for highly multi-modal landscapes with narrow basins.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower'):
            lb, ub = func.lower, func.upper
        else:
            lb, ub = func.bounds.lb, func.bounds.ub
        
        lb, ub = np.array(lb), np.array(ub)
        
        # Initialize state
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        remaining_budget = self.budget - 1
        
        # Adaptive parameters: 1/5th success rule
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        while remaining_budget > 0:
            # Generate candidate
            candidate = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
            
            # Evaluate objective
            current_y = func(candidate)
            remaining_budget -= 1
            
            # Selection and adaptation
            if current_y < best_y:
                best_x, best_y = candidate, current_y
                success_count += 1
            else:
                success_count = 0
            
            # Update sigma using 1/5th rule every few steps
            # Adjust sigma periodically to maintain search efficiency
            if remaining_budget % 5 == 0:
                if success_count > 1:
                    sigma *= 1.2  # Expand if progress is rapid
                else:
                    sigma *= 0.8  # Shrink if progress stalls
            
            # Ensure sigma doesn't collapse to zero
            sigma = np.maximum(sigma, 1e-6)
            
        return best_x, best_y
