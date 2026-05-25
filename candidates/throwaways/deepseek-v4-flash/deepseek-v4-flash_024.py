import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a (1+1)-Evolution Strategy with adaptive step size (1/5 rule)
# for black-box minimization. It is simple, robust, and works across dimensions.
# Search state: A single parent candidate and its objective value, plus a step size
# sigma. The algorithm also maintains a sliding window of recent success/failure
# events to adapt sigma.
# Candidate generation: Each iteration creates one offspring by adding isotropic
# Gaussian noise scaled by sigma to the parent. The noise vector is standard normal.
# Selection and replacement: The parent is replaced by the offspring if the
# offspring has strictly lower (better) objective value (minimization). The best
# point seen so far is always updated when an improvement occurs.
# Adaptation: Step size sigma is adjusted every 'window_size' evaluations based on
# the empirical success rate. If the success rate exceeds 0.2, sigma is increased by
# a factor of 1.2; if below 0.2, sigma is decreased by a factor of 0.8. This
# implements the classic 1/5 success rule, promoting a balanced exploration–
# exploitation trade-off.
# Exploration mechanisms: Mutation with Gaussian noise provides isotropic
# exploration. The step size adaptation helps maintain an appropriate scale of
# perturbation as the search progresses.
# Exploitation mechanisms: The algorithm is elitist – the parent is only replaced by
# better offspring, ensuring that the search remains centered on promising regions.
# Boundary handling: Candidate solutions that fall outside the feasible domain are
# clipped back to the bounds (lower, upper). This is a simple but effective
# repair strategy.
# Budget strategy: The algorithm runs until the evaluation budget is exhausted.
# The initial point consumes one evaluation, and each offspring consumes one.
# The budget is never exceeded because the loop condition checks before generating
# a new candidate.
# Closest known influences: (1+1)-ES with 1/5 rule (Rechenberg, Schwefel).
# Novelty or unusual aspects: None. This is a classic, minimal algorithm chosen for
# compactness and readability.
# Failure modes: The algorithm can stagnate if sigma becomes very small in a region
# far from the global optimum, or if the step size adaptation is too slow. In
# high‑dimensional landscapes with strong ill‑conditioning, the isotropic noise may
# be inefficient. The fixed window size may not be optimal for all problems.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    """
    (1+1)-ES with 1/5 success rule for black-box minimization.

    Parameters
    ----------
    budget : int
        Maximum number of objective function evaluations.
    dim : int
        Dimensionality of the search space.
    """
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def _get_bounds(self, func):
        """Read bounds from func.lower/func.upper or func.bounds.lb/func.bounds.ub."""
        lower = getattr(func, 'lower', None)
        upper = getattr(func, 'upper', None)
        if lower is not None and upper is not None:
            return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        bounds = getattr(func, 'bounds', None)
        if bounds is not None:
            lb = getattr(bounds, 'lb', None)
            ub = getattr(bounds, 'ub', None)
            if lb is not None and ub is not None:
                return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
        raise ValueError("Cannot read bounds from func: expected either "
                         "func.lower/func.upper or func.bounds.lb/func.bounds.ub")

    def __call__(self, func):
        lower, upper = self._get_bounds(func)
        dim = self.dim
        budget = self.budget
        # Initial candidate: uniform random in the domain
        x0 = lower + np.random.uniform(size=dim) * (upper - lower)
        y0 = func(x0)
        best_x = x0.copy()
        best_y = y0
        evals = 1

        # Current parent and step size
        x = x0
        y = y0
        # Initial step size: 20% of the average range
        sigma = 0.2 * np.mean(upper - lower)
        # Window for success rate adaptation
        window_size = max(10, dim)
        successes = []  # sliding list of 0/1

        while evals < budget:
            # Generate offspring
            z = np.random.normal(0.0, 1.0, size=dim)
            x_new = x + sigma * z
            # Clip to bounds
            x_new = np.clip(x_new, lower, upper)
            new_y = func(x_new)
            evals += 1

            # Update best so far
            if new_y < best_y:
                best_y = new_y
                best_x = x_new.copy()

            # Update parent and success record
            if new_y < y:
                x = x_new
                y = new_y
                successes.append(1)
            else:
                successes.append(0)

            # Adapt step size based on success rate over the window
            if len(successes) >= window_size:
                recent = successes[-window_size:]
                success_rate = np.mean(recent)
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                # else unchanged

        return best_x, best_y
