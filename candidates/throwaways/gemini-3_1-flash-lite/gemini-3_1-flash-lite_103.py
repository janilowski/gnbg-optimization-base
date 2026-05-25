# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple yet effective Random Search implementation with a greedy point-tracking mechanism.
# Search state: Keeps track of the current best point and its corresponding objective value.
# Candidate generation: Points are sampled uniformly at random from the defined search space.
# Selection and replacement: The algorithm compares the new point's value to the current best and replaces it if a lower value is found.
# Adaptation: None; the sampling distribution remains uniform throughout the optimization.
# Exploration mechanisms: Global uniform sampling ensures wide coverage of the bounds.
# Exploitation mechanisms: None; relies on pure stochasticity to find the global optimum within the budget.
# Boundary handling: Uses clipping or rejection sampling based on the provided bounds.
# Budget strategy: Iterates exactly until the remaining budget is exhausted.
# Closest known influences: Pure Random Search (Monte Carlo optimization).
# Novelty or unusual aspects: Minimalist design prioritizing robustness and zero dependencies.
# Failure modes: Struggles with high-dimensional problems or objective functions with very sharp, narrow basins of attraction.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initialize the random search algorithm.
        :param budget: Total number of function evaluations allowed.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Execute the minimization using random search.
        :param func: Objective function object with bounds.
        :return: (best_x, best_y)
        """
        # Determine bounds from func object attributes
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        else:
            raise ValueError("Unsupported function interface: No bounds found.")

        best_x = None
        best_y = float('inf')

        # Run random search until the budget is exhausted
        for _ in range(self.budget):
            # Generate a random candidate within the bounds
            candidate_x = np.random.uniform(lb, ub, size=self.dim)
            
            # Evaluate the function
            candidate_y = func(candidate_x)
            
            # Update best if the new point is better
            if candidate_y < best_y:
                best_y = candidate_y
                best_x = candidate_x
                
            # Stop if budget hits 0 (implicit via loop range)
            
        return best_x, best_y
