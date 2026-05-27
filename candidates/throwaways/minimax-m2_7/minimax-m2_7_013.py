# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a classic simulated‑annealing (SA) solver for
#          continuous black‑box minimization.  SA explores the search space by
#          generating random candidate points around the current solution and
#          accepting them either when they improve the objective or with a
#          Metropolis probability that depends on a decreasing temperature.
# Search state: The algorithm maintains a current solution (vector of
#                dimension dim), its objective value, and the best solution
#                seen so far.  A scalar temperature controls the willingness to
#                accept worse moves.
# Candidate generation: A candidate is created by adding a random Gaussian
#                        perturbation to each coordinate.  The perturbation
#                        magnitude is a fixed fraction (10 % by default) of the
#                        allowable range for that coordinate, ensuring the
#                        step size scales naturally with the problem bounds.
# Selection and replacement: If the candidate is better (lower objective) it
#                             replaces the current point.  Otherwise it is
#                             accepted with probability exp(‑Δ/T), where Δ
#                             = candidate‑current objective and T is the current
#                             temperature.
# Adaptation: Temperature follows a geometric cooling schedule that linearly
#              spans from an initial temperature to a very small final
#              temperature over the whole budget.  The step size is kept
#              proportional to the variable ranges; no additional adaptation is
#              performed.
# Exploration mechanisms: Random Gaussian moves provide global exploration,
#                         especially at high temperature.  The Metropolis
#                         acceptance of worse moves allows the algorithm to
#                         escape shallow local minima.
# Exploitation mechanisms: Whenever a candidate improves the current solution,
#                          it is immediately adopted, focusing effort on
#                          promising regions.  The globally best solution is
#                          recorded separately and returned at the end.
# Boundary handling: All generated points are clipped to the problem’s lower
#                    and upper bounds, preventing evaluations outside the
#                    feasible domain.
# Budget strategy: One function evaluation is performed per loop iteration.
#                   The loop runs until the supplied evaluation budget is
#                   exhausted, guaranteeing no more than the allowed calls to
#                   the black‑box function.
# Closest known influences: Classical simulated annealing (Kirkpatrick et al.,
#                           1983) with a geometric cooling schedule and a
#                           simple Gaussian proposal.
# Novelty or unusual aspects: The step size is derived directly from the
#                              variable ranges, making the algorithm
#                              dimension‑agnostic.  The cooling rate is
#                              derived analytically from the budget, so the
#                              temperature progression is automatically tuned
#                              to the number of allowed evaluations.
# Failure modes: If the initial temperature is too low, the search may become
#                greedy and get stuck; if the temperature is too high, too many
#                inferior moves are accepted, wasting the budget.  Likewise, a
#                step size that is too large can cause frequent out‑of‑bounds
#                clipping, while a too‑small step leads to slow progress.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simulated annealing optimizer for continuous black‑box minimization.

    Parameters
    ----------
    budget : int
        Maximum number of objective function evaluations.
    dim : int
        Dimensionality of the search space (number of decision variables).

    Attributes
    ----------
    budget : int
        Evaluation budget.
    dim : int
        Problem dimension.
    initial_temp : float
        Starting temperature for the annealing schedule.
    final_temp : float
        Target temperature after the budget is exhausted.
    cooling_rate : float
        Geometric cooling factor (derived from initial_temp, final_temp, and
        budget).
    step_factor : float
        Fraction of each variable's bound range used as the default step size.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        # Temperature schedule parameters
        self.initial_temp = 1.0
        self.final_temp = 1e-6
        # Compute geometric cooling rate so that temperature linearly decays
        # on a log scale from initial_temp to final_temp over the budget.
        if self.budget > 1:
            self.cooling_rate = (self.final_temp / self.initial_temp) ** (1.0 / (self.budget - 1))
        else:
            # If budget == 1, temperature stays constant (no cooling needed).
            self.cooling_rate = 1.0

        # Step size expressed as a fraction of the variable ranges
        self.step_factor = 0.1

    def __call__(self, func):
        """
        Run the simulated‑annealing optimizer on the given black‑box function.

        Parameters
        ----------
        func : callable
            A black‑box objective function.  It is assumed to accept a 1‑D
            NumPy array of length ``dim`` and return a scalar to be minimized.
            The function object must expose its bounds either as ``func.lower``
            / ``func.upper`` or as ``func.bounds.lb`` / ``func.bounds.ub``.

        Returns
        -------
        best_x : np.ndarray
            The decision vector that achieved the smallest objective value.
        best_y : float
            The corresponding objective value.
        """
        # --------------------------------------------------------------
        # Determine the feasible region (bounds)
        # --------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: unbounded (not typical for benchmarks)
            lower = np.full(self.dim, -1e308, dtype=float)
            upper = np.full(self.dim, 1e308, dtype=float)

        # Ensure lower/upper are arrays of shape (dim,)
        lower = np.atleast_1d(lower)
        upper = np.atleast_1d(upper)
        if lower.shape[0] == 1:
            lower = np.repeat(lower, self.dim)
        if upper.shape[0] == 1:
            upper = np.repeat(upper, self.dim)

        # --------------------------------------------------------------
        # Initialize the first solution (random sampling)
        # --------------------------------------------------------------
        rng = upper - lower
        x = lower + rng * np.random.rand(self.dim)   # current point
        y = func(x)                                   # current objective
        eval_count = 1

        best_x = x.copy()
        best_y = y

        # --------------------------------------------------------------
        # Prepare annealing parameters
        # --------------------------------------------------------------
        T = self.initial_temp
        step_size = self.step_factor * rng   # vector of step sizes per dimension

        # --------------------------------------------------------------
        # Main annealing loop (budget‑constrained)
        # --------------------------------------------------------------
        while eval_count < self.budget:
            # ---- Generate a candidate point ----
            # Gaussian perturbation scaled by step_size
            perturbation = step_size * np.random.randn(self.dim)
            x_candidate = x + perturbation
            # Clip to feasible region
            x_candidate = np.clip(x_candidate, lower, upper)

            # ---- Evaluate candidate ----
            y_candidate = func(x_candidate)
            eval_count += 1

            # ---- Acceptance test (Metropolis criterion) ----
            delta = y_candidate - y   # positive if candidate is worse
            if delta <= 0:
                # Candidate improves the objective
                accept = True
            else:
                # Accept worse candidate with probability exp(-delta/T)
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = np.random.rand() < prob

            if accept:
                x = x_candidate
                y = y_candidate

            # ---- Update global best ----
            if y < best_y:
                best_x = x.copy()
                best_y = y

            # ---- Cool down temperature ----
            T *= self.cooling_rate
            # Optional: adapt step size with temperature (disabled by default)
            # step_size *= (1 - 0.01 * (1 - T / self.initial_temp))

        # --------------------------------------------------------------
        # Return best found solution
        # --------------------------------------------------------------
        return best_x, best_y
