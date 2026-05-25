# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a (1+1)-Evolution Strategy with step-size adaptation using the 1/5 rule. Designed for black-box minimization.
# Search state: Current point x, current step size sigma, iteration count, success history.
# Candidate generation: x' = x + sigma * N(0,I) with coordinate-wise scaling based on bounds range.
# Selection and replacement: Replace x if f(x') < f(x) (elitist). Otherwise reject.
# Adaptation: Every 10 iterations, adjust sigma based on fraction of successful mutations: if >0.2, sigma *= 1.5; if <0.2, sigma /= 1.5.
# Exploration mechanisms: Gaussian random steps with adaptive step size.
# Exploitation mechanisms: Local search around current best; step size shrinks in low success regions.
# Boundary handling: Sticky boundaries: clamp candidate to bounds using clipping.
# Budget strategy: Each iteration uses one function evaluation (plus initial evaluation). Budget exhausted when budget reaches zero.
# Closest known influences: Classic (1+1)-ES with Rechenberg's rule.
# Novelty or unusual aspects: None; straightforward implementation.
# Failure modes: May converge prematurely on multimodal or highly rugged landscapes. Step-size adaptation may fail if success window too short.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds: try func.lower/upper first, then func.bounds.lb/ub
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.array(b.lb, dtype=float)
            ub = np.array(b.ub, dtype=float)
        else:
            raise AttributeError("Cannot locate bounds")

        dim = self.dim
        budget = self.budget
        # Initialization
        # Start at a random point in the domain
        x = lb + np.random.uniform(size=dim) * (ub - lb)
        best_x = x.copy()
        best_y = func(x)
        fevals = 1  # initial evaluation consumed

        # Parameters for (1+1)-ES
        # Step size: initial value as fraction of box width per coordinate
        sigma = 0.2 * (ub - lb)  # vector of step sizes per dimension
        # Adaptation window and counter
        window = 10
        successes = 0
        generations = 0

        # Main loop
        while fevals < budget:
            # Generate candidate by adding Gaussian noise
            # Use sqrt(sigma^2) = sigma, since noise is N(0,I)
            raw_step = np.random.normal(size=dim) * sigma  # element-wise multiplication
            x_candidate = x + raw_step
            # Clamp to bounds (simple clipping)
            x_candidate = np.clip(x_candidate, lb, ub)

            # Evaluate candidate
            y_candidate = func(x_candidate)
            fevals += 1

            # Selection (minimization)
            if y_candidate < best_y:
                best_y = y_candidate
                best_x = x_candidate.copy()
            if y_candidate < func(x):  # compare to current point
                x = x_candidate
                successes += 1

            generations += 1

            # Adapt step size every 'window' generations
            if generations >= window:
                success_rate = successes / window
                if success_rate > 0.2:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma /= 1.5
                # Reset counters
                successes = 0
                generations = 0

        return best_x, best_y
