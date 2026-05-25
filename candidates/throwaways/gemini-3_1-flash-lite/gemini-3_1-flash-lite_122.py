# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (1+1)-Evolution Strategy with adaptive step-size (self-adaptation).
# Search state: Current best point and current step-size (sigma).
# Candidate generation: Gaussian mutation of the current best point.
# Selection and replacement: Deterministic replacement if the new candidate is better.
# Adaptation: One-fifth success rule: increase sigma if success rate is high, decrease otherwise.
# Exploration mechanisms: Global search via Gaussian perturbation.
# Exploitation mechanisms: Local hill-climbing via adaptive mutation step-size.
# Boundary handling: Clipping candidates to the defined objective bounds.
# Budget strategy: Exhaustive loop until eval count matches budget.
# Closest known influences: (1+1)-ES, Rechenberg's 1/5th rule.
# Novelty or unusual aspects: Minimalist implementation of adaptive step-size control.
# Failure modes: Can get stuck in local optima; ineffective for highly deceptive landscapes.
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
        
        # Adaptive step size parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        while evals < self.budget:
            # Generate candidate
            candidate = x + np.random.normal(0, sigma)
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate candidate
            candidate_y = func(candidate)
            evals += 1
            
            # Selection/Replacement
            if candidate_y <= y:
                x, y = candidate, candidate_y
                success_count += 1
            
            # 1/5th Rule adaptation every 10 iterations
            if evals % 10 == 0:
                success_rate = success_count / 10
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                success_count = 0
                
        return x, y
