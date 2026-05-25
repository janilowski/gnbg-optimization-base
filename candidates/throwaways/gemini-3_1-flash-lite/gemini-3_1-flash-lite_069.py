# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (Rastrigin-friendly) random search using progressive shrinkage (shrunk-space search).
# Search state: Maintains the current best point found so far and a shrinking neighborhood radius.
# Candidate generation: Generates candidates via normally distributed perturbations around the current best.
# Selection and replacement: Simple greedy elitism; update current best if a candidate yields a lower objective value.
# Adaptation: The search radius decays geometrically based on the remaining evaluation budget.
# Exploration mechanisms: Initial phase features large radius perturbations to cover the search space.
# Exploitation mechanisms: Final phase narrows the radius to refine the solution.
# Boundary handling: Candidates are clipped to the domain bounds after perturbation.
# Budget strategy: Precisely manages evaluations by allocating one initial sample and then one per iteration until the budget is exhausted.
# Closest known influences: Adaptive Random Search / Simulated Annealing without temperature.
# Novelty or unusual aspects: Minimalist implementation using a single-point state evolution.
# Failure modes: Susceptible to local minima in highly rugged landscapes; may converge too quickly if the decay rate is too high.
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

        # Initial random guess
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        remaining_budget = self.budget - 1

        # Search parameters
        # Initial radius is half the span of the domain
        initial_radius = (ub - lb) * 0.5
        
        # Iterative local search
        for i in range(remaining_budget):
            if remaining_budget <= 0:
                break
                
            # Adaptive radius: shrink based on percentage of budget remaining
            # This allows global exploration early and refinement late
            progress = i / self.budget
            radius = initial_radius * (1.0 - progress)
            
            # Suggest candidate
            perturbation = np.random.normal(0, radius, self.dim)
            candidate_x = np.clip(best_x + perturbation, lb, ub)
            candidate_y = func(candidate_x)
            
            # Elitism: update if better
            if candidate_y < best_y:
                best_y = candidate_y
                best_x = candidate_x
            
            remaining_budget -= 1
            
        return best_x, best_y
