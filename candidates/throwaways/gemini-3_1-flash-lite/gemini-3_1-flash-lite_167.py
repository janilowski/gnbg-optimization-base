# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Local Search algorithm using Adaptive Random Walk (similar to a (1+1)-ES).
# Search state: Maintains a single current best candidate solution (x) and a global step size (sigma).
# Candidate generation: Generates a new candidate by adding Gaussian noise to the current best.
# Selection and replacement: Greedy selection; the new candidate replaces the best if it yields a lower function value.
# Adaptation: The 1/5th success rule is used to adjust sigma: increase if success rate is high, decrease if low.
# Exploration mechanisms: Step size sigma governs exploration scale; initial sigma is proportional to the domain range.
# Exploitation mechanisms: Greedy hill climbing updates improve the current best location.
# Boundary handling: Candidates are clipped to the specified bounds before evaluation.
# Budget strategy: Iteratively samples until the provided function evaluation budget is exhausted.
# Closest known influences: (1+1)-Evolution Strategy.
# Novelty or unusual aspects: Minimalist implementation focusing on robustness in black-box scenarios.
# Failure modes: Susceptible to getting trapped in local minima in highly deceptive surfaces.
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
        
        best_x = np.copy(x)
        best_y = y
        
        # Strategy parameters
        sigma = 0.2 * (ub - lb)
        success_count = 0
        evals = 1
        
        # 1/5th success rule parameters
        c_i, c_d = 1.22, 0.82
        
        while evals < self.budget:
            # Generate candidate
            curr_x = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
            curr_y = func(curr_x)
            evals += 1
            
            # Selection
            if curr_y < best_y:
                best_x = curr_x
                best_y = curr_y
                success_count += 1
            
            # Periodically adapt sigma (every 10 evals to balance exploration/exploitation)
            if evals % 10 == 0:
                success_rate = success_count / 10
                if success_rate > 0.2:
                    sigma *= c_i
                elif success_rate < 0.2:
                    sigma *= c_d
                success_count = 0
                
            # Convergence check: stop if step size becomes negligible
            if np.all(sigma < 1e-10):
                break
                
        return best_x, best_y
