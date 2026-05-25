# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (1+1)-Evolution Strategy with adaptive step-size (1/5th success rule).
# Search state: Maintains a single current best candidate point and a Gaussian mutation step size (sigma).
# Candidate generation: Generates a new candidate by adding spherical Gaussian noise to the current best point.
# Selection and replacement: Uses a greedy selection; the candidate replaces the current point if it yields a lower function value.
# Adaptation: The step size is increased if the success rate is high and decreased if low (1/5th rule) to navigate the landscape.
# Exploration mechanisms: Stochastic mutation provides broad global search driven by the current sigma.
# Exploitation mechanisms: The greedy selection and shrinking sigma focus the search on local optima.
# Boundary handling: Candidates are clipped to the provided search space bounds.
# Budget strategy: Strictly consumes one evaluation per iteration until the budget is exhausted.
# Closest known influences: Rechenberg’s (1+1)-ES.
# Novelty or unusual aspects: Minimalist implementation suitable for diverse black-box continuous functions.
# Failure modes: Can get trapped in local optima; performance degrades on highly rugged landscapes.
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
        curr_x = np.random.uniform(lb, ub)
        curr_y = func(curr_x)
        
        # Hyperparameters for (1+1)-ES
        sigma = 0.2 * (ub - lb)
        success_history = []
        best_x, best_y = curr_x.copy(), curr_y
        
        evals = 1
        while evals < self.budget:
            # Generate candidate
            candidate_x = np.clip(curr_x + np.random.normal(0, sigma), lb, ub)
            candidate_y = func(candidate_x)
            evals += 1
            
            # Selection
            if candidate_y <= curr_y:
                curr_x, curr_y = candidate_x, candidate_y
                success_history.append(1)
                if curr_y < best_y:
                    best_x, best_y = curr_x.copy(), curr_y
            else:
                success_history.append(0)
                
            # Adaptive step-size (1/5th success rule every 10 iterations)
            if len(success_history) >= 10:
                success_rate = sum(success_history) / 10
                if success_rate < 0.2:
                    sigma *= 0.8
                elif success_rate > 0.2:
                    sigma /= 0.8
                success_history = []
                
        return best_x, best_y
