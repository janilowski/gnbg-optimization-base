# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A lightweight (1+1)-evolution strategy with a 1/5‑success rule for step‑size adaptation. It begins with a small random sample to locate an initial good point and then performs iterative Gaussian mutations of the best solution found so far,_accepting improvements and periodically adjusting the mutation spread to balance exploration and exploitation.
# Search state: Keeps the best solution (best_x, best_y), a vector of per‑coordinate step sizes (step_size), an evaluation counter, and counters for successes and evaluations since the last adaptation.
# Candidate generation: Each iteration generates a candidate by adding Gaussian noise (mean 0, variance step_size^2) to the current best point. The noise is independently drawn for each coordinate.
# Selection and replacement: The candidate replaces the current best only if it yields a strictly lower objective value (strict improvement). No elitist archive is maintained beyond the single best.
# Adaptation: Every adapt_interval evaluations (chosen as max(10*dim,200)) the algorithm checks the recent success rate. If the rate exceeds 20%, the step size vector is multiplied by 1.2 (exploration increase); if it is below 20%, the step size is halved (exploitation intensification). The step size is clipped element‑wise to stay within [1e‑6*range, range] where range = upper‑lower.
# Exploration mechanisms: Initial random sampling (≈5% of the budget) provides a first guess, and occasional step‑size increases allow the search to spread across the domain.
# Exploitation mechanisms: When improvements become rare, the step size shrinks, focusing the search around the best point found so far.
# Boundary handling: After mutation, candidate coordinates are clipped to the problem’s lower/upper bounds to ensure feasibility.
# Budget strategy: The algorithm ensures it never exceeds the provided evaluation budget by tracking evaluations and exiting the main loop as soon as the budget is reached.
# Closest known influences: Classic (1+1)-ES with the 1/5‑success rule, similar to simple CMA‑ES variants but without covariance information.
# Novelty or unusual aspects: The use of a per‑coordinate step size vector (scaled by the problem’s range) rather than a single scalar allows anisotropic scaling without maintaining a full covariance matrix, offering a compact compromise between simplicity and adaptivity.
# Failure modes: On highly multi‑modal landscapes the algorithm may converge to a local minimum; the fixed adaptation interval may be too slow to react to rapid changes in landscape curvature when the budget is small.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple (1+1)-evolution strategy with 1/5‑success rule for black‑box minimization.
    """
    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.step_size = None          # will be set in __call__
        self.eval_count = 0

    def __call__(self, func):
        """
        Run the optimization and return the best found solution.

        Parameters
        ----------
        func : callable
            Black‑box objective function. It must accept a 1‑D array_like of length dim
            and return a scalar value.

        Returns
        -------
        best_x : ndarray
            Best candidate solution found.
        best_y : float
            Objective value at best_x.
        """
        # ------------------------------------------------------------------
        # Determine problem bounds (support two common interfaces)
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: unit hypercube if no bounds are exposed
            lower = np.zeros(self.dim, dtype=float)
            upper = np.ones(self.dim, dtype=float)

        # Ensure lower/upper are proper arrays of length dim
        lower = np.broadcast_to(lower, (self.dim,)).copy()
        upper = np.broadcast_to(upper, (self.dim,)).copy()
        range_vec = upper - lower

        # ------------------------------------------------------------------
        # Initialise step size (20% of each coordinate range)
        # ------------------------------------------------------------------
        self.step_size = 0.2 * range_vec

        # ------------------------------------------------------------------
        # Initial random sampling (≈5% of budget, at least one point)
        # ------------------------------------------------------------------
        best_x = None
        best_y = np.inf
        ev = 0
        sample_size = max(1, int(0.05 * self.budget))

        for _ in range(sample_size):
            if ev >= self.budget:
                break
            x = lower + np.random.rand(self.dim) * range_vec
            y = float(func(x))
            ev += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()

        # If budget exhausted after initial sampling, return current best
        if ev >= self.budget:
            return best_x, best_y

        # ------------------------------------------------------------------
        # Main loop: (1+1)-ES with 1/5‑success adaptation
        # ------------------------------------------------------------------
        success_count = 0
        ev_since_adapt = 0
        adapt_interval = max(10 * self.dim, 200)  # evaluations between adaptations

        while ev < self.budget:
            # --- Generate candidate by Gaussian mutation ---
            noise = np.random.randn(self.dim) * self.step_size
            x_candidate = best_x + noise

            # --- Keep candidate within bounds ---
            x_candidate = np.clip(x_candidate, lower, upper)

            # --- Evaluate candidate ---
            y_candidate = float(func(x_candidate))
            ev += 1
            ev_since_adapt += 1

            # --- Accept improvement if strictly better ---
            if y_candidate < best_y:
                best_x = x_candidate
                best_y = y_candidate
                success_count += 1

            # --- Periodic step‑size adaptation (1/5 rule) ---
            if ev_since_adapt % adapt_interval == 0:
                success_rate = success_count / adapt_interval
                if success_rate > 0.2:
                    # Increase exploration
                    self.step_size *= 1.2
                elif success_rate < 0.2:
                    # Increase exploitation
                    self.step_size *= 0.5

                # Clip step size to reasonable bounds (never larger than the range,
                # never smaller than a tiny fraction of the range)
                max_step = range_vec
                min_step = 1e-6 * range_vec
                self.step_size = np.clip(self.step_size, min_step, max_step)

                # Reset counters
                success_count = 0
                ev_since_adapt = 0

        return best_x, best_y
