# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1) Evolution Strategy with a 1/5‑th rule for self‑adapted step size.
#   Starting from a random point, the algorithm repeatedly mutates the current
#   candidate by adding a normally‑distributed offset scaled by sigma. If the
#   mutant improves the objective it replaces the current candidate; the global
#   best solution is updated whenever a better value is found. Every
#   `window` evaluations the recent success rate is computed; sigma is multiplied
#   by 1.2 when the rate exceeds 20 % and by 0.8 otherwise, and kept within
#   `[range/1e6, range*2]`. This balances exploration (large sigma) and
#   exploitation (small sigma).
#
# Search state:
#   - parent (np.ndarray): current candidate solution.
#   - parent_val (float): objective value of the parent.
#   - sigma (float): current mutation strength.
#   - best_x, best_y (np.ndarray, float): best solution observed so far.
#   - evals (int): number of function evaluations performed.
#   - success_history (list[bool]): recent local successes for adaptation.
#
# Candidate generation:
#   child = clip(parent + sigma * N(0, I), lb, ub) where N(0,I) is standard
#   normal in each dimension. Clipping enforces boundary constraints.
#
# Selection and replacement:
#   If child improves the parent’s objective, child becomes the new parent.
#   Global best is updated if child is better than best_y.
#
# Adaptation:
#   After `window = max(10, dim)` evaluations compute success rate.
#   If rate > 0.2 → sigma *= 1.2 else sigma *= 0.8; clamp to [min_sigma,
#   max_sigma] where min_sigma = range/1e6 and max_sigma = range*2.
#
# Exploration mechanisms:
#   Large sigma encourages broad exploration; random normal mutations inject
#   stochasticity across all dimensions.
#
# Exploitation mechanisms:
#   Accepting improving moves focuses the search around better solutions.
#   Decreasing sigma when progress is slow tightens the search radius.
#
# Boundary handling:
#   Child vectors are clipped component‑wise to the problem’s lower/upper
#   bounds obtained from func.lower/.upper or func.bounds.lb/.ub.
#
# Budget strategy:
#   The algorithm performs exactly `budget` evaluations (including the
#   initial evaluation) and never exceeds it.
#
# Closest known influences:
#   Classic (1+1) Evolution Strategy with the 1/5‑th success rule (Rechenberg,
#   1973) and simple self‑adaptation ideas from evolutionary computation.
#
# Novelty or unusual aspects:
#   Minimalist single‑candidate approach; memory usage is O(dim) and
#   computational overhead per evaluation is constant. Suitable for very
#   high budgets and high dimensional spaces without population management.
#
# Failure modes:
#   - If sigma becomes too small, progress may stagnate; however the 1/5‑th
#     rule should recover by increasing sigma when success rate drops.
#   - If sigma grows too large, the search may jump outside informative
#     regions; the upper bound limits this behavior.
#   - For extremely rugged landscapes the simple hill‑climbing character
#     may miss global optima, but the adaptation mechanism provides a
#     reasonable compromise between exploration and exploitation.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    (1+1) Evolution Strategy with 1/5‑th rule for step‑size adaptation.

    Parameters
    ----------
    budget : int
        Maximum number of objective function evaluations.
    dim : int
        Dimensionality of the search space.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimization and return the best found solution.

        Parameters
        ----------
        func : callable
            Objective function to minimize. Must accept a 1‑D NumPy array
            of length `dim` and return a scalar.

        Returns
        -------
        best_x : np.ndarray
            Decision variables of the best solution.
        best_y : float
            Objective value of the best solution.
        """
        if self.budget <= 0:
            raise ValueError("Budget must be positive.")

        # ----- read bounds -------------------------------------------------
        lb, ub = self._get_bounds(func)
        range_vec = ub - lb  # component‑wise range

        # ----- initial candidate -------------------------------------------
        parent = self._random_point(lb, ub)
        parent_val = func(parent)
        evals = 1

        # keep global best
        best_x = parent.copy()
        best_y = parent_val

        # ----- mutation strength --------------------------------------------
        sigma = range_vec / 6.0          # start with ~1/6 of the range
        min_sigma = range_vec / 1e6
        max_sigma = range_vec * 2.0

        # ----- adaptation window -------------------------------------------
        window = max(10, self.dim)       # number of recent attempts for adaptation
        success_history = []              # stores boolean (child better than parent)

        # ----- main loop ----------------------------------------------------
        while evals < self.budget:
            # generate offspring
            child = parent + sigma * np.random.randn(self.dim)
            child = np.clip(child, lb, ub)

            # evaluate
            child_val = func(child)
            evals += 1

            # local competition: does child improve on parent?
            local_success = child_val < parent_val
            if local_success:
                parent = child
                parent_val = child_val
                # update global best if needed
                if child_val < best_y:
                    best_x = child.copy()
                    best_y = child_val

            # record for adaptation
            success_history.append(local_success)

            # adjust sigma when window is full
            if len(success_history) >= window:
                success_rate = sum(success_history) / window
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8

                # keep sigma within safe limits
                sigma = np.clip(sigma, min_sigma, max_sigma)
                success_history = []   # reset window

        return best_x, best_y

    # -----------------------------------------------------------------------
    def _get_bounds(self, func):
        """
        Extract lower and upper bounds from the function object.

        Supports two common interfaces:
          - func.lower / func.upper
          - func.bounds.lb / func.bounds.ub
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, 'bounds'):
            lb = func.bounds.lb
            ub = func.bounds.ub
        else:
            raise AttributeError(
                "func does not expose 'lower'/'upper' or 'bounds.lb'/'bounds.ub'"
            )
        # ensure they are NumPy arrays for vectorised operations
        return np.asarray(lb), np.asarray(ub)

    def _random_point(self, lb, ub):
        """Return a uniform random point inside the box defined by lb/ub."""
        return np.random.uniform(lb, ub)
