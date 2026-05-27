# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (GNBG) minimization algorithm using a population-based strategy with Gaussian perturbations and elitist selection.
# Search state: Maintains a single current best point and its objective value to guide the search.
# Candidate generation: Generates candidates by adding Gaussian noise to the current best point, scaled by a decaying step size.
# Selection and replacement: Simple elitist replacement; if a candidate is better than the current best, it immediately replaces it.
# Adaptation: The step size decays linearly over the course of the budget to transition from exploration to fine-grained exploitation.
# Exploration mechanisms: Initial search is broad due to a larger initial step size (based on domain range).
# Exploitation mechanisms: The step size shrinks as the budget depletes, focusing search on the neighborhood of the current best.
# Boundary handling: Candidates outside the bounds are clipped back into the feasible domain.
# Budget strategy: The number of evaluations is fixed; the algorithm runs exactly until the counter expires.
# Closest known influences: Similar to a (1+1)-ES (Evolution Strategy) with a fixed, deterministic cooling schedule.
# Novelty or unusual aspects: Extremely lightweight implementation with no complex parameter tuning, relying on adaptive scaling of standard deviations.
# Failure modes: Susceptible to local optima entrapment if the initial step size is too small or if the global optimum is in a very narrow basin.
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
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        evals = 1

        # Initial step size (10% of the domain range)
        range_diag = ub - lb
        sigma_start = 0.1 * range_diag

        while evals < self.budget:
            # Calculate current progress factor (1 at start, 0 at end)
            progress = (self.budget - evals) / self.budget
            current_sigma = sigma_start * (progress**2)

            # Generate candidate using Gaussian mutation
            candidate_x = best_x + np.random.normal(0, current_sigma, size=self.dim)
            
            # Boundary handling: Clip to feasible region
            candidate_x = np.clip(candidate_x, lb, ub)
            
            # Evaluate
            candidate_y = func(candidate_x)
            evals += 1
            
            # Elitist replacement
            if candidate_y < best_y:
                best_y = candidate_y
                best_x = candidate_x
            
            # Terminate early if budget met
            if evals >= self.budget:
                break
                
        return best_x, best_y
