# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free Random Search with an exponentially shrinking Gaussian mutation (Symmetric Adaptive Random Search).
# Search state: Stores the best point found so far and the current search radius (sigma).
# Candidate generation: Generates new points by adding Gaussian noise scaled by sigma to the current best point.
# Selection and replacement: Deterministic replacement; if a new point is better than the global best, it replaces it.
# Adaptation: Sigma is halved if no improvement is found after a fixed number of trials, and doubled if an improvement is found, balancing local exploitation and broader search.
# Exploration mechanisms: Initial large sigma (fraction of domain) allows global exploration.
# Exploitation mechanisms: Sigma adaptation focuses search around the current best candidate.
# Boundary handling: Candidates are clipped to the specific bounds provided by the objective function.
# Budget strategy: Exhausts the budget by continuously sampling until calls reach the limit.
# Closest known influences: Adaptive Step Size Random Search.
# Novelty or unusual aspects: Minimalist state tracking with reactive step-size scaling.
# Failure modes: Can get trapped in narrow local minima if the step size shrinks too aggressively.
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
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        evals = 1
        
        # Start with a sigma that covers a fraction of the domain
        sigma = (ub - lb) * 0.2
        
        # Track stagnation to adapt sigma
        stagnant_count = 0
        threshold = max(5, self.dim * 2)
        
        while evals < self.budget:
            # Generate candidate
            candidate = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
            current_y = func(candidate)
            evals += 1
            
            if current_y < best_y:
                # Success: update best and expand search radius
                best_x, best_y = candidate, current_y
                sigma = np.minimum(sigma * 1.1, (ub - lb))
                stagnant_count = 0
            else:
                # Fail: contract search radius if stagnant
                stagnant_count += 1
                if stagnant_count >= threshold:
                    sigma *= 0.5
                    stagnant_count = 0
            
            # Stop if step size is essentially zero
            if np.all(sigma < 1e-12):
                break
                
        return best_x, best_y
