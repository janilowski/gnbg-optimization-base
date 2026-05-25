# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (A)daptive (S)tep size (SAS).
# Search state: Maintains the current best solution and a dynamic global step size (sigma).
# Candidate generation: Generates new candidates by adding Gaussian noise to the current best.
# Selection and replacement: Simple greedy acceptance; if a candidate improves the best, it updates.
# Adaptation: Employs a 1/5th success rule: increases step size if success rate is high, decreases if low.
# Exploration mechanisms: Initialized with a large sigma relative to the search space.
# Exploitation mechanisms: Reduces sigma as the algorithm converges to refine the local minimum.
# Boundary handling: Candidates are clipped to the domain bounds; reflected points are considered if necessary.
# Budget strategy: Iterates until the evaluation count matches the budget; tracks consumption via a counter.
# Closest known influences: Adaptive Random Search, $(1+1)$-Evolution Strategy.
# Novelty or unusual aspects: Minimalist implementation of adaptive sampling without matrix overhead.
# Failure modes: Can get stuck in local optima if the initial exploration does not cover the global basin.
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
        evals = 1

        best_x = np.copy(x)
        best_y = y
        
        # Initial sigma as 20% of the range
        sigma = 0.2 * (ub - lb)
        successes = 0
        
        # Evolution loop
        while evals < self.budget:
            # Generate candidate
            candidate = best_x + np.random.normal(0, sigma)
            # Boundary handling: Clip to stay within feasibility
            candidate = np.clip(candidate, lb, ub)
            
            y_candidate = func(candidate)
            evals += 1
            
            # Acceptance
            if y_candidate < best_y:
                best_y = y_candidate
                best_x = np.copy(candidate)
                successes += 1
            
            # Adapt sigma every 10 iterations (1/5th success rule)
            if evals % 10 == 0:
                if successes / 10 > 0.2:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                successes = 0
                
        return best_x, best_y
