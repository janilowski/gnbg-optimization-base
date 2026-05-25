import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a simple (1+1)-Evolutionary Strategy with a 1/5‑th success rule for step‑size adaptation.
#           The algorithm starts from a random point inside the box constraints, then iteratively perturbs the current
#           solution with a Gaussian offset scaled by a self‑adapted sigma. It maintains the best solution observed
#           and respects the supplied evaluation budget exactly.
#
# Search state:
#   - current_x: the solution currently under consideration (dim‑dimensional numpy array)
#   - current_y: the objective value of current_x
#   - best_x, best_y: the best solution found so far
#   - sigma: current step size (scalar)
#   - budget: remaining number of objective evaluations
#
# Candidate generation:
#   - A candidate is produced as x_new = current_x + sigma * N(0, I) where N(0,I) is a standard normal vector.
#   - The candidate is clipped to the problem's box bounds (lower/upper). If sigma becomes too large, clipping
#     effectively limits exploration, but this is acceptable for a simple baseline.
#
# Selection and replacement:
#   - If the candidate yields a strictly lower objective value, it replaces the current solution and may update
#     the global best (best_x, best_y). Otherwise the current solution remains unchanged.
#
# Adaptation:
#   - The algorithm records whether each step succeeded (i.e., whether the candidate improved the current solution).
#   - A sliding window of the last W = max(10, 3*dim) decisions stores the success flags.
#   - After each evaluation the success rate p = successes / W is computed.
#     - If p > 0.2, sigma is multiplied by 1.1 (step size increase).
#     - If p < 0.2, sigma is multiplied by 0.9 (step size decrease).
#   - sigma is also kept within a reasonable range: at least 1e‑6 times the bound width and at most half the bound width.
#
# Exploration mechanisms:
#   - Gaussian perturbations allow unbounded exploration in principle; the step size controls the typical jump distance.
#   - When the algorithm makes progress, sigma grows, enabling larger moves.
#
# Exploitation mechanisms:
#   - Strict improvement requirement ensures that only better solutions are accepted, focusing search on promising regions.
#   - When progress stalls, sigma shrinks, forcing finer sampling.
#
# Boundary handling:
#   - After each perturbation the candidate vector is clipped component‑wise to the interval [lower, upper].
#   - This keeps all proposals within the feasible domain required by the benchmark.
#
# Budget strategy:
#   - The main loop iterates while the remaining budget is > 0 and performs exactly one evaluation per iteration.
#   - No extra evaluations are performed for adaptation, so the budget is never exceeded.
#
# Closest known influences:
#   - The algorithm is closely related to the simple (1+1)-CMA‑ES and classic (1+1)-ES with the 1/5‑th rule,
#     but it does not maintain a covariance matrix.
#
# Novelty or unusual aspects:
#   - The implementation is intentionally minimalistic, relying only on numpy; it provides a robust baseline across
#     a wide range of dimensionalities without any external dependencies.
#
# Failure modes:
#   - If the budget is smaller than the dimensionality, the algorithm may not have enough evaluations to locate a
#     good solution; in that case the best found will be essentially random.
#   - Highly multi‑modal landscapes may trap the algorithm in local minima, as no mechanism for escaping
#     deep local optima is present.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    """
    Simple (1+1)-Evolutionary Strategy with 1/5‑th rule step‑size adaptation.

    The class follows the required interface:
        __init__(self, budget, dim)
        __call__(self, func) -> (best_x, best_y)
    """

    def __init__(self, budget: int, dim: int):
        """
        Initialize the algorithm.

        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the decision space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the algorithm on the given black‑box function.

        Parameters
        ----------
        func : callable
            Function that receives a numpy array of shape (dim,) and returns a scalar.
            Must expose either ``lower``/``upper`` or ``bounds.lb``/``bounds.ub`` for box constraints.

        Returns
        -------
        best_x : numpy.ndarray
            Best solution found (shape ``(dim,)``).
        best_y : float
            Objective value of the best solution.
        """
        # ------------------------------------------------------------------
        # Obtain problem bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: assume symmetric unit box if bounds are not provided.
            lower = -np.ones(self.dim, dtype=float)
            upper = np.ones(self.dim, dtype=float)

        # Initial step size: a moderate fraction of the overall range.
        sigma = (upper - lower) * 0.1

        # ------------------------------------------------------------------
        # Initialize current and best solutions with a random point.
        # ------------------------------------------------------------------
        rng = np.random  # Use the global RNG that the harness seeds.
        current_x = rng.uniform(lower, upper)
        current_y = float(func(current_x))
        best_x = current_x.copy()
        best_y = current_y

        # ------------------------------------------------------------------
        # Adaptation parameters
        # ------------------------------------------------------------------
        window_size = max(10, 3 * self.dim)  # sliding window for success history
        success_history = []  # list of booleans (True = step succeeded)

        # ------------------------------------------------------------------
        # Main optimization loop
        # ------------------------------------------------------------------
        remaining = self.budget
        while remaining > 0:
            # Generate a candidate by adding a Gaussian perturbation.
            # Clip to bounds to respect the domain.
            candidate = current_x + sigma * rng.randn(self.dim)
            candidate = np.clip(candidate, lower, upper)

            # Evaluate the candidate (consumes one evaluation budget).
            candidate_y = float(func(candidate))
            remaining -= 1

            # Decide whether the step succeeded (improvement).
            succeeded = candidate_y < current_y

            # Update current solution if improvement.
            if succeeded:
                current_x = candidate
                current_y = candidate_y
                # Update global best if this is the best seen.
                if candidate_y < best_y:
                    best_x = candidate.copy()
                    best_y = candidate_y

            # Record success flag for the sliding window.
            success_history.append(succeeded)
            if len(success_history) > window_size:
                success_history.pop(0)

            # ------------------------------------------------------------------
            # 1/5‑th rule step‑size adaptation
            # ------------------------------------------------------------------
            if len(success_history) == window_size:
                success_rate = sum(success_history) / window_size
                if success_rate > 0.2:
                    # Too many successes -> increase step size.
                    sigma *= 1.1
                elif success_rate < 0.2:
                    # Too few successes -> decrease step size.
                    sigma *= 0.9
                # Keep sigma within a sensible range.
                bound_range = upper - lower
                sigma = np.clip(sigma, bound_range * 1e-6, bound_range * 0.5)
                # Reset history after adaptation to keep adaptation reactive.
                success_history.clear()

        return best_x, best_y
