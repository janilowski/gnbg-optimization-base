# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A hybrid optimization approach combining Opposition-Based Learning (OBL) for initialization and an adaptive (1+1) Evolution Strategy for local refinement.
# Search state: Stores the best solution found (best_x), its objective value (best_y), and the current evaluation count.
# Candidate generation: Initial candidates are generated using uniform random sampling followed by their "opposite" points relative to the search space center. Subsequent candidates are generated via Gaussian mutations around the current best point.
# Selection and replacement: Strict elitism is used; a new candidate replaces the current best only if it yields a strictly better objective value.
# Adaptation: The mutation step size (sigma) is scaled by a power-law decay function based on the ratio of the remaining budget to the total budget, ensuring transition from exploration to exploitation.
# Exploration mechanisms: Opposition-style initialization ensures better coverage of the search space, while large initial mutation steps allow the algorithm to jump between basins of attraction.
# Exploitation mechanisms: Local Gaussian search with a shrinking step size allows for precise convergence on the local optimum.
# Boundary handling: All generated candidates are clipped to the hyper-rectangle defined by the lower and upper bounds.
# Budget strategy: A fixed fraction of the budget is allocated for broad initialization, with the remainder used for sequential local improvement. The evaluation count is strictly monitored to prevent exceeding the budget.
# Closest known influences: Opposition-Based Learning (OBL), (1+1) Evolution Strategy, Adaptive Random Search.
# Novelty or unusual aspects: The step size adaptation is tied directly to the remaining budget ratio rather than success rates, providing a predictable convergence schedule.
# Failure modes: On extremely multi-modal landscapes with narrow global optima, the algorithm may converge to a local minimum if the initial samples fail to land in the correct basin.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initializes the algorithm.
        
        :param budget: Total number of function evaluations allowed.
        :param dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Executes the optimization process.
        
        :param func: The objective function to minimize.
        :return: A tuple (best_x, best_y) representing the best found solution.
        """
        # Extract boundary constraints from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Default bounds if none are provided
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        evals_count = 0
        best_x = None
        best_y = float('inf')

        def evaluate(x):
            nonlocal evals_count, best_x, best_y
            if evals_count >= self.budget:
                return best_y
            
            # Ensure the point is within bounds
            x_clipped = np.clip(x, lb, ub)
            y = func(x_clipped)
            evals_count += 1
            
            if y < best_y:
                best_y = y
                best_x = x_clipped.copy()
            return y

        # 1. Opposition-Based Initialization
        # Use roughly 20% of the budget for initial sampling, or at least 2*dim
        init_limit = max(min(self.budget // 5, self.dim * 4), 2)
        
        # Ensure we have at least one valid evaluation if budget > 0
        if self.budget > 0:
            for _ in range(init_limit // 2):
                if evals_count >= self.budget:
                    break
                
                # Random sample
                x_rand = np.random.uniform(lb, ub)
                evaluate(x_rand)
                
                if evals_count >= self.budget:
                    break
                
                # Opposite sample: lb + ub - x
                x_opp = lb + ub - x_rand
                evaluate(x_opp)

        # 2. Local Improvement via Adaptive Random Search
        # Initial step size is 20% of the range
        initial_sigma = 0.2 * (ub - lb)
        
        while evals_count < self.budget:
            # Calculate remaining budget ratio (1.0 down to 0.0)
            rem_ratio = (self.budget - evals_count) / self.budget
            
            # Apply power-law decay to the step size for fine-tuning at the end
            current_sigma = initial_sigma * (rem_ratio ** 1.8)
            
            # Generate a new candidate using Gaussian noise
            noise = np.random.normal(0, 1, self.dim) * current_sigma
            x_candidate = best_x + noise
            
            # Evaluate candidate (clipping happens inside evaluate)
            evaluate(x_candidate)
            
            # If the search space is large or budget is high, occasionally 
            # try a larger jump to escape local optima
            if self.budget > 100 and evals_count % 20 == 0:
                if evals_count < self.budget:
                    x_jump = np.random.uniform(lb, ub)
                    evaluate(x_jump)

        return best_x, best_y
