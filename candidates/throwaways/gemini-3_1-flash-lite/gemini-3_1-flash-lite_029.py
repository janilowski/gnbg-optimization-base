# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free stochastic local search using a shrinking Gaussian mutation strategy (simulated annealing derivative).
# Search state: Maintains a single current best candidate (point and value).
# Candidate generation: Generates new candidates by adding Gaussian noise to the current best, scaled by a step size (sigma).
# Selection and replacement: Greedy selection; if a new candidate is better than the current best, it replaces it.
# Adaptation: The step size 'sigma' is halved if no improvement is found for a specific number of trials to refine the search.
# Exploration mechanisms: Initial large sigma allows global exploration; shrinking sigma forces local convergence.
# Exploitation mechanisms: Local greedy hill-climbing centered around the best-found point.
# Boundary handling: Candidates are clipped to the provided function bounds using numpy.clip.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: (1+1)-ES evolution strategy.
# Novelty or unusual aspects: Extremely compact implementation using adaptive step reduction for robustness.
# Failure modes: Susceptible to local minima in highly rugged landscapes; may converge too slowly if the initial sigma is too small.
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

        # Initialize state
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        evals = 1

        # Search parameters
        sigma = 0.2 * (ub - lb)
        stagnation_limit = max(10, self.dim * 2)
        stagnation_count = 0

        while evals < self.budget:
            # Generate candidate
            candidate = np.clip(best_x + np.random.normal(0, sigma, self.dim), lb, ub)
            current_y = func(candidate)
            evals += 1

            if current_y < best_y:
                # Improvement: accept and reset stagnation
                best_x, best_y = candidate, current_y
                stagnation_count = 0
            else:
                # No improvement: track stagnation
                stagnation_count += 1
                if stagnation_count >= stagnation_limit:
                    # Adapt: shrink step size
                    sigma *= 0.5
                    stagnation_count = 0
                    
                    # Termination condition for precision
                    if np.all(sigma < 1e-9):
                        break

        return best_x, best_y
