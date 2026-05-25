# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (Rastrigin-friendly) random search using an adaptive Gaussian mutation strategy.
# Search state: Tracks the current best point found so far and the adaptive step size (sigma).
# Candidate generation: Generates new points by performing a Gaussian perturbation around the current best candidate.
# Selection and replacement: Greedy selection; if a new point yields a lower objective value, it becomes the new best.
# Adaptation: Employs a simple 1/5th success rule: increases step size if progress is frequent, decreases if rare.
# Exploration mechanisms: Initial global search is wide (determined by bounds), narrowing as the algorithm converges.
# Exploitation mechanisms: Local hill-climbing via Gaussian mutative steps around the incumbent best point.
# Boundary handling: Points are clipped to valid function boundaries after mutation.
# Budget strategy: Uniform distribution of evaluations; the algorithm stops exactly when the budget is reached.
# Closest known influences: A simplified (1+1)-ES (Evolution Strategy).
# Novelty or unusual aspects: Extremely compact implementation designed for black-box environments with minimal overhead.
# Failure modes: Susceptible to getting trapped in local optima for highly multi-modal landscapes with narrow basins of attraction.
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

        # Initialize tracking variables
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        
        # Adaptive step size (sigma) initialized to 10% of the range
        sigma = 0.1 * (ub - lb)
        success_count = 0
        evals_counter = 1
        
        while evals_counter < self.budget:
            # Candidate generation: perturb the best point
            trial_x = np.clip(best_x + np.random.normal(0, sigma), lb, ub)
            trial_y = func(trial_x)
            evals_counter += 1
            
            # Selection: Success-based greedy replacement
            if trial_y < best_y:
                best_y = trial_y
                best_x = trial_x
                success_count += 1
            else:
                success_count = 0
                
            # Adaptation: 1/5th success rule adaptation
            # Every 10 iterations, adjust sigma to influence exploration vs exploitation
            if evals_counter % 10 == 0:
                if success_count > 2:
                    sigma *= 1.2  # Accelerate
                else:
                    sigma *= 0.8  # Refine
                    
            # Ensure sigma doesn't vanish completely
            sigma = np.maximum(sigma, 1e-6 * (ub - lb))
            
        return best_x, best_y
