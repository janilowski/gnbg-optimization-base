# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (n + 1)-point evolutionary strategy (similar to a simple CMA-ES or Simplex variant) using adaptive step sizing.
# Search state: Maintains the current best solution and a diagonal covariance estimate (step size) that adjusts based on successful improvement.
# Candidate generation: Generates candidates via normally distributed mutations centered on the current best point.
# Selection and replacement: Evaluates a small population per generation; replaces the center point if a better candidate is found.
# Adaptation: Employs a simple 1/5th success rule to increase/decrease the mutation scale.
# Exploration mechanisms: Gaussian noise scaled by the step size parameter.
# Exploitation mechanisms: Local search around the current best candidate.
# Boundary handling: Projects candidates back into the feasible region using a clipping method.
# Budget strategy: Iterates until the evaluation budget is exhausted.
# Closest known influences: Evolutionary Strategy (1+lambda)-ES and adaptive random search.
# Novelty or unusual aspects: Extremely compact implementation with reactive step-size adjustment.
# Failure modes: Susceptible to premature convergence in highly multimodal landscapes or deceptive local optima.
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

        # Initialize state
        curr_x = np.random.uniform(lb, ub, self.dim)
        curr_y = func(curr_x)
        evals = 1
        
        best_x = np.copy(curr_x)
        best_y = curr_y
        
        # Hyperparameters
        step_size = 0.2 * (ub - lb)
        pop_size = 4 + int(3 * np.log(self.dim))
        
        while evals < self.budget:
            # Generate offspring
            candidates = []
            for _ in range(min(pop_size, self.budget - evals)):
                # Mutation
                x_new = np.clip(curr_x + np.random.normal(0, step_size), lb, ub)
                y_new = func(x_new)
                evals += 1
                candidates.append((x_new, y_new))
                
                # Update global best
                if y_new < best_y:
                    best_y = y_new
                    best_x = np.copy(x_new)
            
            # Selection: find best of generation
            gen_best_x, gen_best_y = min(candidates, key=lambda p: p[1])
            
            # Adaptation: 1/5th success rule logic
            if gen_best_y < curr_y:
                curr_x, curr_y = gen_best_x, gen_best_y
                step_size *= 1.2  # Accelerate
            else:
                step_size *= 0.8  # Contract
                
            # Convergence check: reset if step size becomes negligible
            if np.all(step_size < 1e-10 * (ub - lb)):
                step_size = 0.2 * (ub - lb)
                curr_x = np.random.uniform(lb, ub, self.dim)
                curr_y = func(curr_x)
                evals += 1
                
        return best_x, best_y
