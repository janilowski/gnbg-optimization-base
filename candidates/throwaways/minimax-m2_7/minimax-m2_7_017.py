# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple hill‑climbing optimiser that generates candidates by perturbing the current best point with a Gaussian step whose size (sigma) adapts according to whether the candidate improves the objective. Periodic random restarts are used to escape local plateaus.
# Search state: Holds the current best solution (best_x), the current step‑size (sigma), the number of evaluations performed, and a stagnation counter.
# Candidate generation: A single candidate is produced as best_x + sigma * N(0, I), where N(0,I) is a standard normal vector. The candidate is clipped to the problem bounds.
# Selection and replacement: If the candidate’s objective value is lower than the best observed so far, the best solution and sigma are updated. Otherwise sigma is reduced.
# Adaptation: sigma is multiplied by 1.2 after a successful improvement and by 0.8 after a failure, clipped to a safe interval relative to the bound range.
# Exploration mechanisms: When no improvement has been observed for dim * 10 iterations, the algorithm restarts from a new random point to explore a different region.
# Exploitation mechanisms: The adaptive sigma controls the local search radius – growing after successes to explore larger steps, shrinking after failures to refine the search.
# Boundary handling: Candidates are projected onto the feasible region using np.clip with the supplied lower and upper bounds.
# Budget strategy: Each loop iteration performs exactly one objective evaluation; the algorithm terminates when the total number of evaluations reaches the supplied budget.
# Closest known influences: Classic hill‑climbing with self‑adaptive step size and restart heuristic, similar to basic random‑restart stochastic search.
# Novelty or unusual aspects: Uses a fixed pair of increase/decrease factors (1.2/0.8) and a simple stagnation threshold instead of more sophisticated adaptive mechanisms.
# Failure modes: On highly multi‑modal problems the simple restart may be insufficient, causing the algorithm to become trapped in local minima, especially in high dimensional spaces when the initial sigma is poorly scaled.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Simple adaptive hill‑climbing optimizer with random restarts."""

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective evaluations allowed.
        dim : int
            Dimensionality of the decision space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimise the given objective function.

        Parameters
        ----------
        func : callable
            Objective function to minimise. Must accept a 1‑D array_like and return a scalar.
            The function is assumed to expose either a ``bounds`` attribute
            (with ``.lb`` and ``.ub``) or ``lower`` / ``upper`` attributes.

        Returns
        -------
        best_x : np.ndarray
            Best solution found.
        best_y : float
            Corresponding objective value.
        """
        # ------------------------------------------------------------------
        # Retrieve problem bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb)
            upper = np.asarray(func.bounds.ub)
        else:
            lower = np.asarray(func.lower)
            upper = np.asarray(func.upper)

        # ------------------------------------------------------------------
        # Initialise search state
        # ------------------------------------------------------------------
        # Random starting point inside the bounds
        x0 = np.random.uniform(lower, upper)
        best_x = x0.copy()
        best_y = func(x0)
        evaluations = 1

        # Initial step‑size (sigma) set to a quarter of the bound range
        rng = upper - lower
        sigma = rng / 4.0

        # Adaptation parameters
        alpha_up = 1.2
        alpha_down = 0.8
        sigma_min = np.min(rng) / 1e6
        sigma_max = np.max(rng) * 2.0

        # Stagnation handling
        max_no_improve = self.dim * 10
        no_improve_count = 0

        # ------------------------------------------------------------------
        # Main optimisation loop
        # ------------------------------------------------------------------
        while evaluations < self.budget:
            # ---- Generate candidate -------------------------------------------------
            direction = np.random.randn(self.dim)
            candidate_x = np.clip(best_x + sigma * direction, lower, upper)
            candidate_y = func(candidate_x)
            evaluations += 1

            # ---- Selection & adaptation -------------------------------------------
            if candidate_y < best_y:
                best_x = candidate_x
                best_y = candidate_y
                sigma = np.clip(sigma * alpha_up, sigma_min, sigma_max)
                no_improve_count = 0
            else:
                sigma = np.clip(sigma * alpha_down, sigma_min, sigma_max)
                no_improve_count += 1

            # ---- Restart after prolonged stagnation --------------------------------
            if no_improve_count >= max_no_improve:
                # Ensure we still have budget for a restart evaluation
                if evaluations >= self.budget:
                    break

                # Random restart point
                x_restart = np.random.uniform(lower, upper)
                best_x = x_restart
                best_y = func(x_restart)
                evaluations += 1

                # Reset sigma and stagnation counter
                sigma = rng / 4.0
                no_improve_count = 0

        return best_x, best_y
