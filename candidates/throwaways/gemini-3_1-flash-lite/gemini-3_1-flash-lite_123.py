# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (A)daptive (S)tep size (ASAS).
# Search state: Maintains the current best solution found so far and an adaptive standard deviation (step size) for proposal generation.
# Candidate generation: Generates new points by sampling from a multivariate normal distribution centered on the current best.
# Selection and replacement: Greedy replacement; the current best is updated only if the new candidate provides a lower objective value.
# Adaptation: The step size is adjusted based on a success rate (1/5th rule): increased if the success rate is high, decreased if low.
# Exploration mechanisms: Global search via initial wide random sampling and step size expansion mechanism.
# Exploitation mechanisms: Local search via Gaussian mutations around the current best candidate.
# Boundary handling: Points are clipped to the feasible domain specified by the objective function.
# Budget strategy: Uniform distribution of evaluations over the entire budget.
# Closest known influences: (1+1)-ES evolution strategy.
# Novelty or unusual aspects: Extremely compact implementation utilizing adaptive step sizing for robustness.
# Failure modes: Can get trapped in sharp local minima in highly deceptive landscapes.
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
        
        # State variables for adaptive step size
        sigma = 0.2 * (ub - lb)
        success_count = 0
        evals = 1
        
        while evals < self.budget:
            # Generate candidate
            candidate = best_x + np.random.normal(0, sigma, self.dim)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            f_val = func(candidate)
            evals += 1
            
            # Selection
            if f_val < best_y:
                best_y = f_val
                best_x = np.copy(candidate)
                success_count += 1
            
            # Adaptive step size adjustment (1/5th success rule)
            if evals % (self.dim * 2) == 0:
                success_rate = success_count / (self.dim * 2)
                if success_rate > 0.2:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                success_count = 0
                
        return best_x, best_y
