# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact, restart‑based randomized hill‑climbing optimizer that samples random starting points,
#          performs adaptive Gaussian perturbations, and keeps the best solution seen within the evaluation budget.
# Search state: Current point, per‑dimension step size, a counter of unsuccessful attempts, and the global best solution.
# Candidate generation: A candidate is created by adding a Gaussian noise vector scaled by the current step size
#                     to the current point. The step size vector is initialised as 10% of the domain interval.
# Selection and replacement: A candidate replaces the current point only if it yields a lower objective value;
#                            otherwise the step size is reduced.
# Adaptation: After a successful improvement the step size is increased by a factor of 1.1; after a failure it is
#            decreased by a factor of 0.9. The algorithm also aborts a local search after a series of failures
#            (≥20) and restarts from a new random point.
# Exploration mechanisms: Random restarts and Gaussian perturbations provide exploration across the search space.
# Exploitation mechanisms: Successful moves are accepted, focusing on local refinement; the gradually shrinking step
#                         size pushes the search toward exploitation as it converges.
# Boundary handling: All points are clipped to the problem's lower and upper bounds after each perturbation.
# Budget strategy: The number of function evaluations is tracked after every call. The outer loop continues while
#                 evaluations remain and each inner local‑search phase stops if the budget is exhausted.
# Closest known influences: Classic Randomized Hill Climbing, Simulated Annealing, and simple Covariance Adaptation
#                          ideas; this implementation stays intentionally simple.
# Novelty or unusual aspects: No novel components; the focus is on readability and robustness across dimensions.
# Failure modes: May become trapped in local minima for highly multi‑modal landscapes; performance depends on the
#                initial step size and adaptation schedule.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple randomized hill‑climbing optimizer with restarts.

    The algorithm samples a random starting point, then performs a local search
    by perturbing the current point with Gaussian noise whose scale adapts based
    on success or failure. When the local search stalls, a new random starting
    point is chosen. The process repeats until the evaluation budget is exhausted.
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the problem (number of decision variables).
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Minimise `func` using the implemented hill‑climbing strategy.

        Parameters
        ----------
        func : callable
            Objective function to be minimized. It must accept a 1‑D array
            of length `dim` and return a scalar value.

        Returns
        -------
        best_x : ndarray
            The decision vector that achieved the lowest observed objective value.
        best_y : float
            The corresponding objective value.
        """
        # Retrieve problem bounds (lower, upper).
        lower, upper = self._get_bounds(func)

        # Initialise tracking variables.
        best_x = None
        best_y = np.inf
        evals = 0

        # Outer loop: keep restarting until the budget is exhausted.
        while evals < self.budget:
            # 1) Sample a random starting point within the bounds.
            x = lower + (upper - lower) * np.random.rand(self.dim)
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()

            # 2) Begin local search from this point.
            # Initial step size vector: 10% of the domain width per dimension.
            step = (upper - lower) * 0.1
            attempts = 0
            no_improve = 0
            max_attempts_per_restart = 200  # safeguard against infinite loops.

            # Local search loop.
            while (evals < self.budget and
                   attempts < max_attempts_per_restart and
                   np.any(step > 1e-8)):
                attempts += 1

                # Generate a candidate by adding Gaussian noise scaled by `step`.
                candidate = x + step * np.random.randn(self.dim)
                candidate = np.clip(candidate, lower, upper)

                y_candidate = func(candidate)
                evals += 1

                if y_candidate < y:
                    # Improvement found: move to the candidate.
                    x = candidate
                    y = y_candidate

                    # Increase step size to encourage broader exploration.
                    step = step * 1.1
                    no_improve = 0

                    # Update global best if needed.
                    if y < best_y:
                        best_y = y
                        best_x = x.copy()
                else:
                    # No improvement: shrink step size and count failure.
                    step = step * 0.9
                    no_improve += 1
                    if no_improve >= 20:
                        # Too many consecutive failures → abort local search.
                        break

        return best_x, best_y

    def _get_bounds(self, func):
        """
        Extract lower and upper bounds from the function object.

        The function may store bounds as attributes `lower`/`upper` or as a
        `bounds` object with `lb`/`ub` attributes. If none are found, a very
        wide default range is used.

        Parameters
        ----------
        func : callable
            Objective function whose bounds are to be extracted.

        Returns
        -------
        lower, upper : ndarray
            1‑D arrays of length `dim` representing the lower and upper bounds.
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.atleast_1d(func.lower)
            upper = np.atleast_1d(func.upper)
        elif hasattr(func, 'bounds'):
            lower = np.atleast_1d(func.bounds.lb)
            upper = np.atleast_1d(func.bounds.ub)
        else:
            # Fallback: huge bounds (should rarely be needed).
            lower = np.full(self.dim, -1e6)
            upper = np.full(self.dim, 1e6)

        # Ensure the bounds are of the correct length.
        if lower.shape[0] < self.dim:
            lower = np.broadcast_to(lower, (self.dim,))
        if upper.shape[0] < self.dim:
            upper = np.broadcast_to(upper, (self.dim,))

        return lower, upper
