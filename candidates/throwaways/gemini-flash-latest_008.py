# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A hybrid stochastic optimizer combining global uniform sampling with adaptive local perturbations around the incumbent.
# Search state: Stores the best solution found (best_x, best_y) and the current search scale.
# Candidate generation: Alternates between generating a fully uniform random sample (exploration) and a Gaussian mutation of the incumbent (exploitation).
# Selection and replacement: Simple greedy selection (elitist); the incumbent is replaced only if a new point yields a strictly lower objective value.
# Adaptation: The mutation step size (standard deviation) decays exponentially as the budget is consumed, transitioning from coarse to fine-grained local search.
# Exploration mechanisms: A fixed proportion of evaluations is dedicated to uniform sampling across the entire search space to avoid premature convergence.
# Exploitation mechanisms: Gaussian perturbations centered on the current best point, with a diminishing radius to refine the solution.
# Boundary handling: All generated candidates are clipped to the hyper-rectangular bounds defined by the problem instance.
# Budget strategy: Strict iteration count tracking to ensure the function is called exactly up to the budget limit.
# Closest known influences: Random Search, (1+1)-Evolution Strategy with fixed schedule step-size adaptation.
# Novelty or unusual aspects: Uses a simple ratio-based switch between global and local search without complex success-history tracking.
# Failure modes: High-dimensional highly-multimodal landscapes may require more sophisticated population-based diversity mechanisms than this single-point approach.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    A robust black-box minimizer using a hybrid of global uniform exploration 
    and incumbent-centered Gaussian exploitation with a cooling schedule.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.exploration_ratio = 0.2  # 20% of budget spent on global uniform sampling

    def __call__(self, func):
        # Determine bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback if bounds are not explicitly provided in expected formats
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        range_width = ub - lb
        
        # Initial evaluation: center of the search space
        best_x = lb + 0.5 * range_width
        best_y = func(best_x)
        eval_count = 1

        # Optimization loop
        while eval_count < self.budget:
            # Calculate progress for step-size adaptation (0.0 to 1.0)
            progress = eval_count / self.budget
            
            # Decide between exploration (global) and exploitation (local)
            if np.random.rand() < self.exploration_ratio:
                # Global Exploration: Uniformly random point
                candidate = lb + np.random.rand(self.dim) * range_width
            else:
                # Local Exploitation: Perturb the current best solution
                # Step size decays from 0.5 of the range to 0.0001
                sigma = 0.5 * np.exp(-7.0 * progress)
                candidate = best_x + np.random.normal(0, sigma, self.dim) * range_width
            
            # Boundary handling: ensure candidate is within [lb, ub]
            candidate = np.clip(candidate, lb, ub)
            
            # Evaluation
            y = func(candidate)
            eval_count += 1
            
            # Greedy replacement
            if y < best_y:
                best_y = y
                best_x = candidate

        return best_x, best_y

# The harness provides the seed via np.random.seed().
# This implementation uses numpy for efficiency and readability.
