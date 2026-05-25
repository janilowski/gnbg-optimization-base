import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implementation of a (1+1) Evolution Strategy (ES) with self-adaptive step size using the 1/5 success rule for black-box minimization.
# Search state: A single candidate solution (best_x) and a step-size parameter (sigma) that controls mutation magnitude.
# Candidate generation: Perturb best_x with Gaussian noise scaled by sigma. Boundary clipping ensures feasibility.
# Selection and replacement: The offspring replaces the parent if it yields a lower objective value (greedy elitism).
# Adaptation: Step-size sigma is adjusted every N evaluations (N=10) based on the success rate over that window. If success rate > 0.2, sigma is increased; if < 0.2, decreased.
# Exploration mechanism: Large sigma values at early stages enable global search; the adaptive rule prevents premature convergence.
# Exploitation mechanism: Small sigma values allow fine local refinement when success rate declines.
# Boundary handling: Out-of-bounds candidates are clipped to the feasible domain.
# Budget strategy: Each evaluation is counted; the algorithm stops exactly when the budget is exhausted.
# Closest known influences: Rechenberg's (1+1)-ES with 1/5 rule, a classic evolutionary algorithm for continuous optimization.
# Novelty or unusual aspects: None; this is a straightforward implementation of a well-known method.
# Failure modes: May stagnate on functions with highly non-separable or ill-conditioned landscapes due to isotropic mutation; can get stuck in local optima; step-size adaptation may be slow for high-dimensional problems.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # --- Extract bounds -------------------------------------------------
        # Support both 'func.lower/upper' and 'func.bounds.lb/ub'
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):  # e.g., scipy's bounds object
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot determine bounds from the objective function.")

        # Ensure they are 1‑D and the length matches self.dim
        lb = lb.flatten()
        ub = ub.flatten()
        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError("Bounds dimension does not match self.dim")

        # --- Initialization -------------------------------------------------
        # Uniform random initial point inside bounds
        best_x = lb + (ub - lb) * np.random.rand(self.dim)
        best_y = func(best_x)
        remaining = self.budget - 1  # initial evaluation already spent

        # Initial step size: 1/6 of the domain width per dimension (global sigma)
        sigma = np.mean(ub - lb) / 6.0   # scalar, isotropic

        # --- (1+1)-ES loop -------------------------------------------------
        window_size = 10          # generations used for the 1/5 rule
        success_window = []      # list of boolean successes to compute rate
        gen_count = 0

        while remaining > 0:
            # Generate offspring
            candidate = best_x + np.random.normal(0, sigma, size=self.dim)
            # Clamp to bounds (simple, ensures feasibility)
            candidate = np.clip(candidate, lb, ub)

            # Evaluate
            candidate_y = func(candidate)
            remaining -= 1

            # Selection (greedy)
            if candidate_y < best_y:
                best_x = candidate
                best_y = candidate_y
                success_window.append(True)
            else:
                success_window.append(False)

            gen_count += 1

            # Every "window_size" generations apply the 1/5 rule
            if gen_count == window_size:
                success_rate = np.mean(success_window)
                if success_rate > 0.2:
                    sigma *= 1.2          # increase mutation step
                elif success_rate < 0.2:
                    sigma *= 0.8          # decrease mutation step
                # Reset counters for next window
                success_window = []
                gen_count = 0

            # Ensure sigma does not collapse to zero or explode
            sigma = max(sigma, 1e-12)
            # Also avoid sigma being larger than the domain width (sensible cap)
            sigma = min(sigma, 0.5 * np.mean(ub - lb))

        # Return the best found solution and its value
        return best_x, best_y
