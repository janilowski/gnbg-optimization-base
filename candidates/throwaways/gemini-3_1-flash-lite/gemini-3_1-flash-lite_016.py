# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (GNBG) adaptive search algorithm using a restricted-covariance CMA-ES variant with localized perturbation.
# Search state: Maintains a current mean vector and a step-size (sigma) that scales with the progress of the search.
# Candidate generation: Generates samples using a multivariate normal distribution centered at the current best point.
# Selection and replacement: Uses an elitist selection strategy, updating the mean only if a superior point is found.
# Adaptation: Employs a simplified step-size adaptation (1/5th success rule) to regulate exploration intensity.
# Exploration mechanisms: Initial search is broad with a large sigma; global coverage is ensured by the random initialization and wide initial variance.
# Exploitation mechanisms: Localized convergence is achieved by shrinking sigma once improvement halts, focusing on local basin refinement.
# Boundary handling: Projects candidate solutions back into the feasible region by clamping values to the defined lower and upper bounds.
# Budget strategy: Iterative step-based approach that monitors the remaining function evaluation budget strictly.
# Closest known influences: Inspired by CMA-ES and the (1+1)-ES evolutionary strategy.
# Novelty or unusual aspects: Extremely compact implementation that ignores covariance matrix updates in favor of isotropic perturbations to ensure stability across varying dimensions.
# Failure modes: May get stuck in sharp local minima if the basin is significantly smaller than the initial sigma, or struggle with extremely high-dimensional, highly non-separable landscapes.
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
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        remaining_budget = self.budget - 1
        
        # Step size starts at 1/5th of the range
        sigma = 0.2 * (ub - lb)
        
        # Iterative refinement ((1+1)-ES strategy)
        while remaining_budget > 0:
            # Generate candidate
            candidate = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
            
            # Evaluate
            current_y = func(candidate)
            remaining_budget -= 1
            
            # Success: Move mean, increase step size (for faster movement)
            if current_y < best_y:
                best_x = candidate
                best_y = current_y
                sigma *= 1.2
            # Failure: Shrink step size (for finer exploitation)
            else:
                sigma *= 0.95
            
            # Reset sigma if it becomes too small to move effectively
            if np.all(sigma < 1e-10 * (ub - lb)):
                sigma = 0.1 * (ub - lb)
                
        return best_x, best_y
