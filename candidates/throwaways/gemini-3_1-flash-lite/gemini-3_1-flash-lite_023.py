# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple yet effective Random Search implementation with a modest adaptive component for the objective function.
# Search state: Tracks the best found solution (best_x, best_y).
# Candidate generation: Generates uniformly distributed random vectors within the defined bounds.
# Selection and replacement: Updates best_x and best_y if a new candidate provides a lower function value.
# Adaptation: The search is purely stochastic and does not adapt parameters based on function landscape.
# Exploration mechanisms: Global uniform sampling explores the entire space evenly.
# Exploitation mechanisms: None, relies entirely on dense random sampling within the budget.
# Boundary handling: Clamps candidates to the function bounds if necessary, though direct uniform sampling is centered on bounds.
# Budget strategy: Exhausts the provided budget exactly by performing sequential function evaluations.
# Closest known influences: Pure Random Search.
# Novelty or unusual aspects: Minimalist design ensuring robustness and avoiding bias towards specific basins.
# Failure modes: Performs poorly on high-dimensional landscapes with sharp, isolated narrow minima.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initialize the random search algorithm.
        :param budget: Total number of allowed function evaluations.
        :param dim: Dimensionality of the problem space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the minimization process using random sampling.
        :param func: The objective function to minimize.
        :return: (best_x, best_y)
        """
        # Determine bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        else:
            raise ValueError("Could not determine bounds from function.")

        best_x = None
        best_y = float('inf')

        # Sequential sampling within the budget
        for _ in range(self.budget):
            # Generate a candidate vector using uniform distribution across the bounding box
            x = np.random.uniform(lb, ub, size=self.dim)
            
            # Evaluate the function
            y = func(x)
            
            # Update best found solution
            if y < best_y:
                best_y = y
                best_x = x.copy()
        
        # Ensure we return at least one valid point if budget > 0
        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)
            best_y = func(best_x)
            
        return best_x, best_y
