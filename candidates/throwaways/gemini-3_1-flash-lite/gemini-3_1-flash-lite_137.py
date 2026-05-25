# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple yet effective Random Search implementation with a greedy update mechanism for black-box minimization.
# Search state: Stores the best solution candidate (x) and its corresponding function value (y) found so far.
# Candidate generation: Generates uniformly distributed random points within the specified hyper-rectangular bounds.
# Selection and replacement: Replaces the incumbent if a new evaluation yields a strictly lower objective value.
# Adaptation: None; the sampling distribution remains uniform across the search space.
# Exploration mechanisms: Global random sampling ensures uniform coverage of the search space.
# Exploitation mechanisms: Greedy selection retains the best seen point to move toward the global minimum.
# Boundary handling: Uses clips to ensure points remain within [lb, ub].
# Budget strategy: Iterates exactly until the budget is exhausted, evaluating one point per iteration.
# Closest known influences: Pure Random Search.
# Novelty or unusual aspects: Extremely lightweight and robust against non-convex, disconnected, or noisy landscapes.
# Failure modes: Inefficient in high dimensions or for functions with narrow global basins requiring fine-tuned local search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds safely from common industry interfaces
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        else:
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)

        best_x = None
        best_y = float('inf')

        # Run random search until budget is exhausted
        for _ in range(self.budget):
            # Sample candidate point uniformly within bounds
            candidate = np.random.uniform(lb, ub, self.dim)
            
            # Evaluate objective
            y = func(candidate)
            
            # Greedy update if improvement is found
            if y < best_y:
                best_y = y
                best_x = candidate
                
        return best_x, best_y
