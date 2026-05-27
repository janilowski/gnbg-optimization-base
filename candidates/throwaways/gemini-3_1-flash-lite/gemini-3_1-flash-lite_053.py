# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (1+1)-Evolution Strategy with adaptive step-size (1/5th success rule).
# Search state: Maintains a single current best point and the global step size sigma.
# Candidate generation: Gaussian mutation of the current best point.
# Selection and replacement: Greedy selection (elitism); if the candidate is better, it replaces the current point.
# Adaptation: Sigma is increased after successful iterations and decreased after failures to maintain a ~20% success rate.
# Exploration mechanisms: Adaptive global mutation scale ensures coverage across the search landscape.
# Exploitation mechanisms: Elitist selection focuses on the best found point.
# Boundary handling: Resampling or clipping candidates to ensure they stay within bounds.
# Budget strategy: Uniform step-by-step consumption until exactly zero remaining.
# Closest known influences: (1+1)-ES, CMA-ES (simplified).
# Novelty or unusual aspects: Minimalist implementation using the 1/5th rule for simplicity.
# Failure modes: Susceptible to local optima and premature convergence in highly rugged landscapes.
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
        x = np.random.uniform(lb, ub, self.dim)
        y = func(x)
        self.budget -= 1
        
        best_x, best_y = np.copy(x), y
        sigma = 0.2 * (ub - lb)
        success_count = 0
        gen = 0
        
        while self.budget > 0:
            # Generate candidate
            candidate = x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            candidate_y = func(candidate)
            self.budget -= 1
            
            # Selection
            if candidate_y < y:
                x, y = candidate, candidate_y
                success_count += 1
                if y < best_y:
                    best_x, best_y = np.copy(x), y
            
            # Adaptive step size (1/5th success rule)
            gen += 1
            if gen >= 5:
                if success_count / gen > 0.2:
                    sigma *= 1.2
                elif success_count / gen < 0.2:
                    sigma *= 0.8
                gen = 0
                success_count = 0
                
            # Restart if stuck (sigma too small)
            if np.all(sigma < 1e-10 * (ub - lb)):
                sigma = 0.2 * (ub - lb)
                x = np.random.uniform(lb, ub, self.dim)
                y = func(x)
                self.budget -= 1
                
        return best_x, best_y
