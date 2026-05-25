# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (n + lambda) evolution strategy variant.
# Search state: Tracks the best point found so far as the current mean of the population.
# Candidate generation: Generates new candidates by sampling from a multivariate normal distribution centered at the best known point with a decaying step size.
# Selection and replacement: Uses an elitist strategy where the best point replaces the parent if it improves the objective.
# Adaptation: Step size decays linearly as a function of remaining budget to transition from exploration to fine-grained exploitation.
# Exploration mechanisms: Initial large variance (sigma) and random normal sampling.
# Exploitation mechanisms: Elitism and gradual reduction of the search radius.
# Boundary handling: Clamping candidates to the [lower, upper] bounds.
# Budget strategy: Divides the budget into batch-based iterations, ensuring the total limit is never exceeded.
# Closest known influences: Simplified (1+lambda)-ES or basic CMA-ES without covariance matrix adaptation.
# Novelty or unusual aspects: Minimalist implementation using only NumPy; robust fallback for bounds extraction.
# Failure modes: May get trapped in local minima in highly deceptive, non-convex landscapes.
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

        center = np.random.uniform(lb, ub)
        best_y = func(center)
        best_x = center.copy()
        
        # Hyperparameters
        evals_remaining = self.budget - 1
        pop_size = 10
        
        # Initial sigma is 1/5th of the search space diagonal
        sigma_start = 0.2 * (ub - lb)
        
        iteration = 0
        while evals_remaining > 0:
            # Decay sigma: starts exploration, ends exploitation
            progress = iteration / (self.budget / pop_size)
            sigma = sigma_start * (1.0 - min(progress, 0.95))
            
            # Generate candidates
            candidates = []
            for _ in range(min(pop_size, evals_remaining)):
                candidate = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
                candidates.append(candidate)
            
            # Evaluate candidates
            for cand in candidates:
                y = func(cand)
                evals_remaining -= 1
                if y < best_y:
                    best_y = y
                    best_x = cand.copy()
            
            # Move center towards best found
            iteration += 1
            if evals_remaining <= 0:
                break
                
        return best_x, best_y
