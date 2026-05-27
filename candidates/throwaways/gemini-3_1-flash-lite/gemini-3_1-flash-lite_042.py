# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (1+1)-Evolution Strategy with adaptive step-size (1/5th success rule).
# Search state: Maintains a single candidate solution 'x' and its function value 'y'.
# Candidate generation: Gaussian mutation added to the current best candidate.
# Selection and replacement: Greedy selection; if the new candidate is better, replace the current best.
# Adaptation: Step size is increased if success rate is high (success > 20%) and decreased if low (success < 20%).
# Exploration mechanisms: Stochastic Gaussian perturbations allow for global exploration.
# Exploitation mechanisms: Local hill-climbing via adaptive step size contraction.
# Boundary handling: Clipping the mutated candidate vector to defined lower/upper bounds.
# Budget strategy: Iterative loop until evaluation count exactly matches budget.
# Closest known influences: (1+1)-CMA-ES/Evolution Strategy.
# Novelty or unusual aspects: Extremely compact and minimal memory footprint.
# Failure modes: Can get trapped in local optima in highly deceptive landscapes or high-dimensional needle-in-haystack scenarios.
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
        evals = 1
        
        # Step size adaptation parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        while evals < self.budget:
            # Generate candidate
            candidate = np.clip(x + np.random.normal(0, sigma), lb, ub)
            
            # Evaluate
            candidate_y = func(candidate)
            evals += 1
            
            # Selection
            if candidate_y <= y:
                x, y = candidate, candidate_y
                success_count += 1
            
            # 1/5th success rule adaptation every 10 iterations
            if evals % 10 == 0:
                success_rate = success_count / 10
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                success_count = 0
                
        return x, y
