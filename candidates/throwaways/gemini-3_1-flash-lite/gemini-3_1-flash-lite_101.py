# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (Rastrigin/GNBG-friendly) local search using a shrinking-radius adaptive random walk.
# Search state: Current best position and its objective value.
# Candidate generation: Gaussian sampling around the current best with an adaptive step size.
# Selection and replacement: Greedy: replace current best if a candidate improves the objective.
# Adaptation: Step size doubles on success and halves on failure (1/5th rule inspired).
# Exploration mechanisms: Initial large-scale sampling and dynamic step size expansion.
# Exploitation mechanisms: Local hill climbing via shrinking the search radius around the current best.
# Boundary handling: Clamping candidates to the provided box constraints.
# Budget strategy: Iterative loop until the number of calls reaches the allocated budget.
# Closest known influences: Adaptive Random Search / Solis-Pots algorithm.
# Novelty or unusual aspects: Minimalist implementation focusing on stability for complex landscapes.
# Failure modes: Can get trapped in deep local minima if the global optimum is not reachable via local improvement.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds safely
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Initialize randomly within bounds
        curr_x = np.random.uniform(lb, ub, self.dim)
        curr_y = func(curr_x)
        
        best_x, best_y = curr_x.copy(), curr_y
        
        # Hyperparameters for adaptive search
        step_size = (ub - lb) * 0.1
        evals = 1
        
        while evals < self.budget:
            # Generate candidate
            candidate = curr_x + np.random.normal(0, step_size, self.dim)
            candidate = np.clip(candidate, lb, ub)
            
            cand_y = func(candidate)
            evals += 1
            
            # Greedy update with adaptive step size logic
            if cand_y < curr_y:
                curr_x, curr_y = candidate, cand_y
                step_size *= 1.2  # Accelerate towards descent
                
                if curr_y < best_y:
                    best_x, best_y = curr_x.copy(), curr_y
            else:
                step_size *= 0.5  # Refine around local neighborhood
                
            # Restart if step size vanishes to maintain exploration
            if np.all(step_size < 1e-9):
                step_size = (ub - lb) * 0.05
                
        return best_x, best_y
