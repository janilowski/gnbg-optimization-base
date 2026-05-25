# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (D)ecreasing (S)tep-size.
# Search state: Tracks the best point found so far and the current radius of search.
# Candidate generation: Random Gaussian perturbations around the best point found.
# Selection and replacement: Greedy update; point is replaced if function value improves.
# Adaptation: Step-size (radius) is halved if no improvement occurs over a fixed number of trials.
# Exploration mechanisms: Initial large radius allows global coverage; decays to local refinement.
# Exploitation mechanisms: Concentrates sampling near the current best candidate.
# Boundary handling: Points are clipped to function bounds before evaluation.
# Budget strategy: Iterates until the evaluation count reaches the budget.
# Closest known influences: Adaptive Random Search / Stochastic Hill Climbing.
# Novelty or unusual aspects: Minimalist implementation using only standard library and numpy.
# Failure modes: Can get trapped in local optima; ineffective on highly multi-modal landscapes with narrow basins.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialize: Start at center of search space
        best_x = (lb + ub) / 2.0
        best_y = func(best_x)
        self.eval_count = 1

        radius = (ub - lb) / 4.0
        patience = self.dim * 5
        stagnation = 0

        # Run until budget exhausted
        while self.eval_count < self.budget:
            # Generate candidate via Gaussian perturbation
            z = np.random.normal(0, 1, self.dim)
            candidate = best_x + z * radius
            
            # Clip to bounds
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            y = func(candidate)
            self.eval_count += 1
            
            if y < best_y:
                # Success: update best and reset stagnation
                best_y = y
                best_x = candidate
                stagnation = 0
            else:
                # Failure: increase stagnation count
                stagnation += 1
                
            # Adapt step size if necessary
            if stagnation >= patience:
                radius *= 0.5
                stagnation = 0
                
            # Stop if radius is negligible
            if np.all(radius < 1e-12):
                break
                
        return best_x, best_y
