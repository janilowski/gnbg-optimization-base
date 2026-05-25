# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of the $(1+1)$-Evolution Strategy with adaptive step-size (1/5th success rule).
# Search state: Maintains the current 'best_x' and 'best_y' found so far and the current mutation step size 'sigma'.
# Candidate generation: Generates a single candidate point by adding Gaussian noise to the current best_x, scaled by sigma.
# Selection and replacement: Greedy selection; if the candidate evaluates better than the current best, it becomes the new best.
# Adaptation: Employs the 1/5th success rule to update sigma: increase if success rate is higher than 0.2, decrease otherwise.
# Exploration mechanisms: Gaussian mutation provides local search, while initial aggressive sigma allows covering the space.
# Exploitation mechanisms: Local hill-climbing via successful mutations reducing the radius towards the optimum.
# Boundary handling: Candidates are clipped to the provided lower and upper bounds.
# Budget strategy: A simple loop that continues until the evaluation count reaches the budget.
# Closest known influences: Rechenberg's (1+1)-ES.
# Novelty or unusual aspects: Minimalist state management suitable for black-box configuration.
# Failure modes: Can get trapped in local optima; performance degrades on highly rugged landscapes.
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

        # Initial solution: center of bounds
        x = (lb + ub) / 2.0
        y = func(x)
        evals = 1

        # State initialization
        sigma = (ub - lb) * 0.1
        success_count = 0
        
        # Adaptive step size parameters
        t = 0
        
        while evals < self.budget:
            t += 1
            # Generate candidate
            z = x + np.random.normal(0, sigma, size=self.dim)
            z = np.clip(z, lb, ub)
            
            # Evaluate
            y_cand = func(z)
            evals += 1
            
            # Selection
            if y_cand <= y:
                x, y = z, y_cand
                success_count += 1
            
            # 1/5th success rule adaptation every 10 generations
            if t % 10 == 0:
                success_rate = success_count / 10
                if success_rate < 0.2:
                    sigma *= 0.8
                elif success_rate > 0.2:
                    sigma *= 1.2
                success_count = 0
                
        return x, y
