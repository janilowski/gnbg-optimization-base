# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk (Self-Adaptive Step Size).
# Search state: Tracks the current best solution and the current step size (sigma).
# Candidate generation: Generates a candidate point by perturbing the current best using multivariate normal noise scaled by sigma.
# Selection and replacement: Greedy selection; updates current best if the objective value improves.
# Adaptation: Employs the "1/5th success rule": if a move is successful, increase sigma; otherwise, decrease it to refine search.
# Exploration mechanisms: Initial large sigma allows global exploration, which shrinks as the search progresses.
# Exploitation mechanisms: Local refinement via decreasing step size and greedy updates.
# Boundary handling: Candidates are clipped to the search space defined by the function bounds.
# Budget strategy: Exhausts the total evaluation budget sequentially.
# Closest known influences: (1+1)-Evolution Strategy.
# Novelty or unusual aspects: Minimalist implementation of adaptive step-size control without population management.
# Failure modes: Susceptible to local optima; step size may shrink too quickly in high-dimensional or rugged landscapes.
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

        # Initialization
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        
        best_x = np.copy(x)
        best_y = y
        
        # Initial step size (10% of the range)
        sigma = 0.1 * (ub - lb)
        
        evals = 1
        
        # 1/5th success rule parameters
        success_count = 0
        
        while evals < self.budget:
            # Generate candidate
            noise = np.random.normal(0, sigma, self.dim)
            candidate = np.clip(best_x + noise, lb, ub)
            
            # Evaluate
            f_val = func(candidate)
            evals += 1
            
            # Selection
            if f_val < best_y:
                best_x = np.copy(candidate)
                best_y = f_val
                success_count += 1
                # Increase exploration if consistently successful
                sigma *= 1.1
            else:
                # Tighten search
                sigma *= 0.95
            
            # Periodically adapt sigma based on success rate (every 10 evals)
            if evals % 10 == 0:
                if success_count / 10 > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                success_count = 0
                
        return best_x, best_y
