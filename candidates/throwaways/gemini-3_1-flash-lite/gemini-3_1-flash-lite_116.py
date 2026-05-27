# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (COBYLA-inspired) local search using a simplex-based trust region approach.
# Search state: Maintains a current best point and a neighborhood radius (trust region) that shrinks over time.
# Candidate generation: Generates candidates via random perturbations within the current trust region and reflection steps.
# Selection and replacement: Greedy update; keeps the best observed point and centers the search around it if a better point is found.
# Adaptation: The trust region radius decreases linearly as the budget is consumed to shift from exploration to exploitation.
# Exploration mechanisms: Initial large trust region and random sampling; reflections allow escaping small local optima.
# Exploitation mechanisms: Local contraction of the trust region and adaptive step sizing.
# Boundary handling: Points are clipped to valid bounds; reflected points that hit boundaries are projected back.
# Budget strategy: Uniform distribution of evaluations; the loop terminates when the budget is reached.
# Closest known influences: Pattern Search and basic trust-region simplicial methods.
# Novelty or unusual aspects: Minimalist implementation using a simple decaying radius schedule.
# Failure modes: Can get trapped in sharp local minima or fail on highly non-convex surfaces with long, narrow ridges.
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
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        evals = 1
        
        # Adaptive parameters
        radius = 0.5 * (ub - lb)
        
        while evals < self.budget:
            # Decay radius based on remaining budget (linear schedule)
            progress = evals / self.budget
            current_radius = radius * (1.0 - progress)
            
            # Generate candidate via random perturbation
            direction = np.random.normal(0, 1, self.dim)
            step = direction * current_radius
            candidate_x = np.clip(best_x + step, lb, ub)
            
            # Evaluate
            candidate_y = func(candidate_x)
            evals += 1
            
            # Greedy update
            if candidate_y < best_y:
                best_y = candidate_y
                best_x = candidate_x
            else:
                # Use reflection if improvement not found
                reflected_x = np.clip(best_x - 0.5 * step, lb, ub)
                reflected_y = func(reflected_x)
                evals += 1
                
                if reflected_y < best_y:
                    best_y = reflected_y
                    best_x = reflected_x
            
            # Early break if budget reached inside loop
            if evals >= self.budget:
                break
                
        return best_x, best_y
