# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk (Hill Climbing with Shrinking Steps) designed for robust black-box minimization.
# Search state: Tracks the current best-found point (x_best) and the current step size (sigma).
# Candidate generation: Generates a candidate point by perturbing the current best using a multivariate normal distribution scaled by sigma.
# Selection and replacement: Greedy update rule: if the candidate performs better than the current best, the candidate becomes the new current best.
# Adaptation: The step size 'sigma' is adapted using the one-fifth success rule: it increases when the success rate is high and decreases when low, facilitating faster convergence.
# Exploration mechanisms: Initial large step sizes allow broad exploration, which naturally transitions to fine-grained exploitation as sigma shrinks.
# Exploitation mechanisms: Local search around the current optimum using shrinking neighborhoods.
# Boundary handling: Candidates are clipped to the provided box constraints, ensuring feasibility.
# Budget strategy: Iteratively evaluates points until 'budget' is exhausted.
# Closest known influences: (1+1)-ES (Evolution Strategy) and basic hill climbing.
# Novelty or unusual aspects: Extremely compact implementation that dynamically manages exploration-exploitation trade-offs without complex memory structures.
# Failure modes: Susceptible to getting trapped in narrow local minima in highly multimodal landscapes due to the lack of a global restart mechanism.
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
        
        # Initialize
        budget_left = self.budget
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        budget_left -= 1
        
        # Initial sigma (step size) as 1/5th of the range
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        # Search loop
        while budget_left > 0:
            # Generate candidate
            candidate = best_x + np.random.normal(0, sigma, self.dim)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            y = func(candidate)
            budget_left -= 1
            
            # Selection
            if y < best_y:
                best_x, best_y = candidate, y
                success_count += 1
            else:
                success_count -= 0.25 # Penalize stagnation
            
            # Adaptation of sigma (1/5th rule logic)
            # Adjust step size every few iterations to balance exploration/exploitation
            if (self.budget - budget_left) % 5 == 0:
                if success_count > 1:
                    sigma *= 1.2
                elif success_count < 0:
                    sigma *= 0.8
                success_count = 0
                # Ensure sigma doesn't collapse to zero
                sigma = np.maximum(sigma, 1e-9 * (ub - lb))
                
        return best_x, best_y
