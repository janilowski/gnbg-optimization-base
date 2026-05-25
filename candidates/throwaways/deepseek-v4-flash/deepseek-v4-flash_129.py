import numpy as np
from collections import deque

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1)-Evolution Strategy with step-size adaptation and occasional uniform sampling.
# Search state: Single current best solution vector (best_x) and its objective value (best_y).
# A global step size (sigma) controls the magnitude of perturbations.
# Candidate generation: With 95% probability, candidate = best_x + sigma * N(0,1).
# With 5% probability, candidate = uniform random point in the domain to encourage exploration.
# Selection and replacement: Greedy – if candidate_y < best_y, replace best_x and best_y.
# Adaptation: Step size is updated after every evaluation using a sliding window of the last L successes (L = max(5,dim)).
# The success rate over the window is compared to the target 1/5. If above, sigma *= 1.2; if below, sigma *= 0.85.
# Sigma is clipped to avoid extreme values.
# Exploration mechanisms: Isotropic Gaussian perturbations plus occasional uniform restarts.
# Exploitation mechanisms: The greedy update and shrinking step size (when success rate is low) focus search near the current best.
# Boundary handling: Candidate coordinates are clipped to the lower and upper bounds.
# Budget strategy: Every evaluation is counted; the loop stops exactly when the budget is exhausted.
# Closest known influences: Standard (1+1)-ES with Rechenberg’s 1/5 success rule.
# Novelty or unusual aspects: Simple uniform restarts (5% chance) to escape local optima.
# Failure modes: May struggle on non-separable, ill-conditioned, or highly multimodal problems due to isotropic step size.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            bounds = func.bounds
            lb = np.asarray(bounds.lb, dtype=float)
            ub = np.asarray(bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide lower/upper or bounds.lb/ub")

        dim = self.dim
        evals = 0

        # Initial point
        best_x = np.random.uniform(lb, ub, size=dim)
        best_y = func(best_x)
        evals = 1

        # Step size
        sigma = 0.2 * (ub - lb)  # per-dimension scale, but we use a single global sigma
        # For simplicity we use a scalar sigma; for per-dim adaptation one would use a vector.
        # Using the mean scale across dimensions
        sigma = np.mean(sigma)

        # Success tracking with sliding window (size L)
        L = max(5, dim)
        recent_successes = deque(maxlen=L)
        adapt_interval = L

        # Main loop
        while evals < self.budget:
            # Generate candidate
            if np.random.random() < 0.05:
                # Uniform restart
                c = np.random.uniform(lb, ub, size=dim)
            else:
                # Gaussian perturbation
                c = best_x + sigma * np.random.randn(dim)
            # Clip to bounds
            c = np.clip(c, lb, ub)

            # Evaluate
            c_y = func(c)
            evals += 1

            # Greedy selection
            if c_y < best_y:
                best_x = c[:]
                best_y = c_y
                success = 1
            else:
                success = 0

            # Update sliding window
            recent_successes.append(success)

            # Adapt sigma once we have enough data
            if evals >= L:
                success_rate = sum(recent_successes) / L
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.85
                # Protect sigma from extreme values
                sigma = np.clip(sigma, 1e-12 * np.mean(ub - lb), 0.5 * np.mean(ub - lb))

        return best_x, best_y
