# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (RMA-ES inspired) evolutionary strategy using adaptive mutation for continuous optimization.
# Search state: Maintains a current mean vector representing the center of the search distribution.
# Candidate generation: Generates a population of samples by adding Gaussian noise to the current mean.
# Selection and replacement: Selects the best performing individuals from the current generation to update the mean.
# Adaptation: Employs a step-size (sigma) that shrinks over the budget to focus on exploitation.
# Exploration mechanisms: Initial large sigma encourages global exploration.
# Exploitation mechanisms: Weighted averaging of top performers focuses the mean towards the local optimum.
# Boundary handling: Clamping to provided bounds at each iteration/sampling step.
# Budget strategy: Iterative loop that terminates exactly when the evaluation count reaches the budget.
# Closest known influences: Simplified Evolution Strategy (ES) / CMA-ES.
# Novelty or unusual aspects: Minimalist implementation without complex covariance matrix tracking.
# Failure modes: Susceptible to local optima in multi-modal landscapes and convergence speed depends heavily on initial sigma.
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
        pop_size = int(4 + np.floor(3 * np.log(self.dim)))
        num_parents = max(1, pop_size // 2)
        mean = np.random.uniform(lb, ub)
        sigma = 0.3 * (ub - lb)
        
        best_x = None
        best_y = float('inf')
        evals = 0

        # Optimization loop
        while evals < self.budget:
            # Generate population
            pop_x = []
            pop_y = []
            
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                
                # Sample and clamp
                x = np.clip(mean + sigma * np.random.randn(self.dim), lb, ub)
                y = func(x)
                evals += 1
                
                pop_x.append(x)
                pop_y.append(y)
                
                # Update global best
                if y < best_y:
                    best_y = y
                    best_x = x.copy()

            if evals >= self.budget:
                break
                
            # Selection: sort offspring
            pop_y = np.array(pop_y)
            idx = np.argsort(pop_y)
            
            # Update mean using top parents
            parents_x = np.array([pop_x[i] for i in idx[:num_parents]])
            mean = np.mean(parents_x, axis=0)
            
            # Adaptation: reduce sigma slowly
            sigma *= 0.95 

        return best_x, best_y
