# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A derivative-free (n + 1)-point evolutionary strategy.
# Search state: A current best solution and a population of small-step perturbations.
# Candidate generation: Generates candidates using Gaussian noise added to the current best.
# Selection and replacement: Simple elitist (greedy) update rule.
# Adaptation: A simple adaptive step size (1/5 rule) to maintain search progress.
# Exploration mechanisms: Gaussian mutation with adaptive step size.
# Exploitation mechanisms: Elitist selection preserves the best local point.
# Boundary handling: Projections (clipping) to feasible bounds.
# Budget strategy: Iterative step-by-step evaluation until the budget is exhausted.
# Closest known influences: (1+lambda)-ES and simple hill-climbing.
# Novelty or unusual aspects: Minimalist implementation aiming for robustness in low-to-medium dimensions.
# Failure modes: Can get stuck in local optima; likely inefficient for extremely high-dimensional spaces.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.step_size = 0.3
        self.success_count = 0
        self.total_count = 0

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialize: pick best random point
        best_x = np.random.uniform(lb, ub, self.dim)
        best_y = func(best_x)
        evals_used = 1

        # Search loop
        while evals_used < self.budget:
            # Generate candidate
            noise = np.random.normal(0, self.step_size, self.dim)
            candidate = np.clip(best_x + noise, lb, ub)
            
            # Evaluate
            y = func(candidate)
            evals_used += 1
            
            # Selection
            if y < best_y:
                best_x, best_y = candidate, y
                self.success_count += 1
            
            self.total_count += 1

            # Adaptive step size (1/5 rule)
            if self.total_count >= 10:
                ratio = self.success_count / self.total_count
                if ratio > 0.2:
                    self.step_size *= 1.1
                else:
                    self.step_size /= 1.1
                self.success_count = 0
                self.total_count = 0

        return best_x, best_y
