# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (R)andom (S)earch with (C)entered (A)daptive (S)tep-size (Scale).
# Search state: Stores the best solution found so far and the current adaptive search radius.
# Candidate generation: Generates candidates via isotropic Gaussian perturbations around the best-known solution.
# Selection and replacement: Standard greedy replacement: if a candidate improves the objective, update the best solution.
# Adaptation: The step size shrinks when no improvement is found after a cooling period; grows slightly upon success.
# Exploration mechanisms: Initial global search with a large radius; transitions to local exploitation.
# Boundary handling: Clamps candidates to the search space defined by function bounds.
# Budget strategy: Uniformly consumes the evaluation budget across the search duration.
# Closest known influences: Adaptive Random Search (ARS).
# Novelty or unusual aspects: Minimalist implementation using only NumPy without external dependencies.
# Failure modes: Can get stuck in local minima in high-dimensional landscapes; poor performance on highly deceptive functions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """Initializes the adaptive random search."""
        self.budget = budget
        self.dim = dim
        self.sigma = 0.5  # Initial search radius

    def __call__(self, func):
        """Minimizes the given objective function using adaptive random search."""
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        remaining_budget = self.budget - 1
        
        # Adaptive parameters
        no_improve_count = 0
        patience = max(10, self.dim * 2)

        while remaining_budget > 0:
            # Generate candidate using Gaussian mutation
            # Scale sigma relative to the domain range
            radius = self.sigma * (ub - lb)
            candidate = best_x + np.random.normal(0, radius, self.dim)
            
            # Boundary constraint enforcement
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluate
            y = func(candidate)
            remaining_budget -= 1
            
            # Selection
            if y < best_y:
                best_x, best_y = candidate, y
                self.sigma = min(1.0, self.sigma * 1.1)  # Expand search
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Adaptation: Shrink if stagnant
            if no_improve_count >= patience:
                self.sigma *= 0.5
                no_improve_count = 0
                
            # Convergence check: stop if search radius is microscopically small
            if np.all(self.sigma * (ub - lb) < 1e-12):
                break
                
        return best_x, best_y
