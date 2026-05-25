# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (n + 4)-multi-start local search algorithm leveraging Gaussian perturbations.
# Search state: Keeps track of the global best solution found so far.
# Candidate generation: Generates samples using isotropic Gaussian distributions centered on the best-known point.
# Selection and replacement: Greedy updates; replaces the current best if a sample yields a lower function value.
# Adaptation: Employs a simple 1/5th success rule heuristic to adapt the Gaussian step size (sigma).
# Exploration mechanisms: Starts with a large step size relative to the domain to explore the search space.
# Exploitation mechanisms: Gradually shrinks sigma as the search converges to refine the local minimum.
# Boundary handling: Projects candidate points back into the feasible domain using clipping.
# Budget strategy: Divides the total budget into a series of iterations; terminates exactly when budget is exhausted.
# Closest known influences: A simplified (1+1)-ES (Evolution Strategy) with adaptive step size.
# Novelty or unusual aspects: Minimalist, robust, and requires no parameter tuning other than the initial step size.
# Failure modes: Susceptible to getting trapped in narrow local minima in highly deceptive, multi-modal landscapes.
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
        
        # Initialize search state
        curr_x = np.random.uniform(lb, ub)
        curr_y = func(curr_x)
        self.budget -= 1
        
        best_x, best_y = curr_x.copy(), curr_y
        
        # Adaptive parameters: sigma is the step size
        sigma = 0.2 * (ub - lb)
        success_count = 0
        
        while self.budget > 0:
            # Generate candidate
            step = np.random.normal(0, sigma)
            candidate = np.clip(best_x + step, lb, ub)
            
            # Evaluate
            y = func(candidate)
            self.budget -= 1
            
            # Selection
            if y < best_y:
                best_x, best_y = candidate, y
                success_count += 1
            
            # Adaptation (1/5th success rule)
            # Adjust sigma periodically to maintain efficient convergence
            if (self.budget % 10) == 0:
                if success_count / 10 > 0.2:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                success_count = 0
                
            # Random restart if exhausted or trapped (if budget allows)
            if self.budget > 10 and np.all(sigma < 1e-7 * (ub - lb)):
                best_x = np.random.uniform(lb, ub)
                best_y = func(best_x)
                self.budget -= 1
                sigma = 0.2 * (ub - lb)
                
        return best_x, best_y
