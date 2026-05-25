# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A multi‑start random search combined with a lightweight coordinate‑descent local improver.
# Search state: The algorithm maintains the best solution found (best_x, best_y) and the current point used in the local search together with per‑coordinate step sizes.
# Candidate generation: New trial points are drawn uniformly at random inside the box constraints.
# Selection and replacement: After each function evaluation the best solution is updated if an improvement is observed.
# Adaptation: The step size for each coordinate shrinks (is halved) when a move in either direction does not improve the objective, providing a simple form of adaptation.
# Exploration mechanisms: Random restarts ensure coverage of the whole domain; uniform sampling provides broad exploration.
# Exploitation mechanisms: Coordinate‑wise hill‑climbing with decreasing step sizes focuses the search around the current candidate.
# Boundary handling: Proposed moves are clamped to the lower and upper bounds so that evaluations stay within the feasible region.
# Budget strategy: The total number of function evaluations is strictly limited by the provided budget; the algorithm interleaves random sampling and local search while counting every call.
# Closest known influences: Multi‑start random search and coordinate search are classic, simple approaches used in many black‑box optimization testbeds.
# Novelty or unusual aspects: Per‑coordinate step adaptation with a halving rule is a straightforward way to balance exploration and exploitation without any parameter tuning.
# Failure modes: For highly multi‑modal landscapes the algorithm may get trapped in local minima, especially if the budget is small relative to the dimension.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple black‑box optimizer for the GNBG benchmark.

    The optimizer performs a series of random restarts, each followed by a
    coordinate‑wise hill‑climbing phase with an adaptive step size.
    All function evaluations are counted against the supplied budget.
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the problem (i.e., number of decision variables).
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimizer on the given objective function.

        Parameters
        ----------
        func : callable
            A black‑box function that accepts a NumPy array of shape (dim,)
            and returns a scalar (the objective value).  The function must
            expose its bounds either via ``func.lower``/``func.upper`` or
            ``func.bounds.lb``/``func.bounds.ub``.

        Returns
        -------
        best_x : np.ndarray
            The best (lowest) solution found.
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------------
        # 1. Determine the search bounds (box constraints).
        # ------------------------------------------------------------------
        lb, ub = self._get_bounds(func)

        # ------------------------------------------------------------------
        # 2. Initial random point – also serves as first candidate.
        # ------------------------------------------------------------------
        x = lb + np.random.rand(self.dim) * (ub - lb)
        y = func(x)
        evals = 1

        best_x = x.copy()
        best_y = y

        # ------------------------------------------------------------------
        # 3. Main loop – keep generating random restarts until the budget is exhausted.
        # ------------------------------------------------------------------
        while evals < self.budget:
            # ----- 3a. Random restart -------------------------------------------------
            x = lb + np.random.rand(self.dim) * (ub - lb)
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()

            # ----- 3b. Local improvement (coordinate search) -------------------------
            # Initial step size per coordinate (10 % of the range)
            step = 0.1 * (ub - lb)

            # Limit local work to avoid consuming the whole budget too quickly.
            # We allow at most dim * 20 evaluations for this local phase,
            # but also respect the remaining budget.
            max_local = min(self.budget - evals, self.dim * 20)
            local_evals = 0

            while local_evals < max_local and evals + local_evals < self.budget:
                # Pick a random coordinate to try moving.
                i = np.random.randint(self.dim)

                # If the range for this coordinate is zero, skip it.
                if step[i] == 0:
                    local_evals += 1
                    continue

                # ----- 3b‑i. Positive direction ------------------------------------
                x_pos = x.copy()
                x_pos[i] = min(x[i] + step[i], ub[i])
                y_pos = func(x_pos)
                local_evals += 1
                if y_pos < y:
                    x = x_pos
                    y = y_pos
                    if y < best_y:
                        best_y = y
                        best_x = x.copy()
                    continue  # accept the move and try another coordinate

                # ----- 3b‑ii. Negative direction -----------------------------------
                x_neg = x.copy()
                x_neg[i] = max(x[i] - step[i], lb[i])
                y_neg = func(x_neg)
                local_evals += 1
                if y_neg < y:
                    x = x_neg
                    y = y_neg
                    if y < best_y:
                        best_y = y
                        best_x = x.copy()
                    continue  # accept the move

                # ----- 3b‑iii. No improvement – shrink step for this coordinate ----
                step[i] *= 0.5
                # Prevent the step from shrinking indefinitely.
                if step[i] < 1e-10 * (ub[i] - lb[i] + 1e-12):
                    step[i] = 0.0

            # Account for all local evaluations.
            evals += local_evals

        return best_x, best_y

    # ----------------------------------------------------------------------
    # Helper: read bounds from the function object.
    # ----------------------------------------------------------------------
    def _get_bounds(self, func):
        """
        Extract lower and upper bounds from ``func``.

        Supports two common conventions:
            * ``func.lower`` / ``func.upper``  (attributes)
            * ``func.bounds.lb`` / ``func.bounds.ub``  (attribute with sub‑attributes)

        If neither is present, the unit hyper‑cube [0,1]^dim is assumed.
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: unit hyper‑cube.
            lb = np.zeros(self.dim, dtype=float)
            ub = np.ones(self.dim, dtype=float)

        # Ensure the shape matches the dimensionality.
        if lb.shape != (self.dim,):
            lb = np.broadcast_to(lb, self.dim).copy()
        if ub.shape != (self.dim,):
            ub = np.broadcast_to(ub, self.dim).copy()

        return lb, ub
