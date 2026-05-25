# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1)-Evolution Strategy with step-size adaptation for black-box minimization.
# Search state: A single candidate solution (best found so far) and an adaptive step-size (sigma).
# Candidate generation: Add isotropic Gaussian noise scaled by sigma to the current best solution.
# Selection and replacement: Deterministic – the new candidate replaces the current best only if it improves the objective.
# Adaptation: Rechenberg's 1/5 success rule – after each block of iterations (length dim) the step-size is increased if the success rate exceeds 1/5, otherwise decreased.
# Exploration mechanisms: Large initial step-size relative to variable bounds; adaptation keeps sigma from collapsing too quickly.
# Exploitation mechanisms: Decreasing step-size when improvements are rare, focusing search near the current best.
# Boundary handling: Clipping (projection) of infeasible coordinates into the hyper-rectangle defined by bounds.
# Budget strategy: Uses exactly the allowed number of function evaluations, stopping when the budget is exhausted.
# Closest known influences: Classic (1+1)-ES with Rechenberg's rule (Rechenberg, 1973).
# Novelty or unusual aspects: None – deliberately kept simple, robust, and interpretable.
# Failure modes: Might converge prematurely to a local optimum; step-size can shrink too much in rugged landscapes; does not handle constraints other than box bounds.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective function must provide bounds via .lower/.upper or .bounds.lb/.bounds.ub")

        # Problem dimension
        dim = self.dim
        if dim != len(lb):
            # Safety: use func dimension if given, else fallback
            # Actually dim from __init__ should match, but ensure
            dim = len(lb)

        # Initial step-size (scale relative to bounds)
        domain_range = ub - lb
        # Avoid zero-range dimensions
        domain_range = np.maximum(domain_range, 1e-6)
        sigma = 0.2 * np.mean(domain_range)  # single scalar step-size

        # Initial random point inside bounds
        x = lb + np.random.random(dim) * domain_range
        best_y = func(x)
        evals = 1
        best_x = x.copy()

        # State tracking for 1/5 rule
        total_steps = 0
        success_steps = 0

        # Main loop: run until budget exhausted
        while evals < self.budget:
            # Generate candidate: isotropic Gaussian mutation
            # Clip to bounds after mutation
            noise = sigma * np.random.randn(dim)
            x_new = best_x + noise
            x_new = np.clip(x_new, lb, ub)

            # Evaluate if budget remains
            if evals >= self.budget:
                break
            y_new = func(x_new)
            evals += 1

            # Selection: accept if better (minimization)
            if y_new < best_y:
                best_x = x_new.copy()
                best_y = y_new
                success_steps += 1

            total_steps += 1

            # Apply Rechenberg's 1/5 rule every dim steps
            if total_steps >= dim:
                success_rate = success_steps / total_steps
                if success_rate > 0.2:
                    sigma *= 1.2  # increase exploration
                else:
                    sigma *= 0.9  # focus on exploitation
                # Reset counters
                total_steps = 0
                success_steps = 0

                # Prevent sigma from becoming too small or too large
                sigma = max(sigma, 1e-10 * np.mean(domain_range))
                sigma = min(sigma, 0.5 * np.mean(domain_range))

        return best_x, best_y
