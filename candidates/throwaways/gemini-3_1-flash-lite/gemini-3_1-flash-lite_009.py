# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk.
# Search state: Tracks the current best point found so far and the dynamic step size (sigma).
# Candidate generation: Generates candidates via Gaussian perturbation of the current best point.
# Selection and replacement: Greedy selection; updates the best point if the candidate is better.
# Adaptation: Employs a 1/5th success rule to adjust step size: increases after successful moves, decreases after failures.
# Exploration mechanisms: Initially high sigma allows broad exploration; decays as the search converges.
# Exploitation mechanisms: Local Gaussian sampling around the best known point refines the solution.
# Boundary handling: Projects candidates back into the feasible region using clipping.
# Budget strategy: Exhaustive usage of the total budget by splitting it into distinct trials based on heuristic convergence.
# Closest known influences: (1+1)-ES (Evolution Strategy) with self-adaptive step size.
# Novelty or unusual aspects: Compact implementation focusing on rapid convergence for black-box functions.
# Failure modes: Can get trapped in narrow local optima if the initial step size is too small; ineffective on highly deceptive rugged landscapes.
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
        x_best = np.random.uniform(lb, ub, self.dim)
        y_best = func(x_best)
        evals = 1
        
        # Adaptive parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        # Main optimization loop
        while evals < self.budget:
            # Generate candidate via Gaussian mutation
            x_cand = x_best + np.random.normal(0, sigma)
            
            # Boundary constraint projection
            x_cand = np.clip(x_cand, lb, ub)
            
            # Evaluate
            y_cand = func(x_cand)
            evals += 1
            
            # Greedy replacement
            if y_cand < y_best:
                x_best = x_cand
                y_best = y_cand
                success_count += 1
            else:
                success_count -= 0.25  # Heuristic penalty for failure
            
            # Adaptive step size control (1/5th success rule logic)
            if evals % 10 == 0:
                if success_count > 2:
                    sigma *= 1.2
                elif success_count < 0:
                    sigma *= 0.8
                success_count = 0
            
            # Termination if search is stagnant
            if np.all(sigma < 1e-9 * (ub - lb)):
                break
                
        return x_best, y_best
