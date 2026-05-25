# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a simple 1+1 Evolution Strategy with a 1/5‑style step‑size adaptation (random‑direction hill climbing).
# Search state: The current best point (x), a step size (sigma), and a counter of function evaluations.
# Candidate generation: For each iteration a random direction is drawn from a standard normal distribution, normalized, scaled by the current step size, and added to the current point.
# Selection and replacement: The candidate replaces the current point only if it yields a lower (better) objective value.
# Adaptation: After a successful move the step size is multiplied by a factor > 1 (1.1) to encourage larger steps; after a failure it is multiplied by a factor < 1 (0.9) to reduce step size. Bounds keep sigma within a safe range.
# Exploration mechanisms: Random Gaussian directions provide isotropic exploration; step size controls the scale.
# Exploitation mechanisms: By only accepting improvements the search focuses on moving toward better solutions.
# Boundary handling: Candidate points are clipped to the problem’s bounds before evaluation.
# Budget strategy: The algorithm counts each evaluation and stops as soon as the counter reaches the provided budget, guaranteeing the budget is never exceeded.
# Closest known influences: Classic 1+1 ES with the 1/5 success rule, and simple hill‑climbing with random directions.
# Novelty or unusual aspects: Step size adaptation uses simple multiplicative factors rather than a formal success ratio, yielding a compact yet effective mechanism.
# Failure modes: May converge prematurely on highly non‑convex landscapes and can exhibit oscillations if the objective is noisy.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple black‑box minimization algorithm using a 1+1 evolution strategy
    with random direction sampling and adaptive step size.
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the problem (number of decision variables).
        """
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        """
        Minimize `func` within the provided evaluation budget.

        Parameters
        ----------
        func : callable
            Objective function to be minimized. It must accept a 1‑D numpy array
            of length `dim` and return a scalar value.

        Returns
        -------
        best_x : numpy.ndarray
            Decision vector that achieved the smallest observed value.
        best_y : float
            Objective value corresponding to ``best_x``.
        """
        # ------------------------------------------------------------------
        # Obtain problem bounds (lower and upper) from the function object.
        # ------------------------------------------------------------------
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb, dtype=float)
                ub = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                # Fallback to a wide default box if bounds are not provided.
                lb = -100.0 * np.ones(self.dim)
                ub = 100.0 * np.ones(self.dim)

        # Ensure lb/ub are proper 1‑D arrays.
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        # ------------------------------------------------------------------
        # Initialise search state.
        # ------------------------------------------------------------------
        # Random start point uniformly inside the bounds.
        x_current = np.random.uniform(lb, ub)

        # Evaluate the initial point (counts as one evaluation).
        best_x = x_current.copy()
        best_y = func(x_current)
        evals = 1

        # If the budget is zero we cannot evaluate anything; return what we have.
        if self.budget <= 0:
            return best_x, best_y

        # Initial step size (sigma) set to 5% of the range per dimension.
        span = ub - lb
        sigma = 0.05 * span

        # Safeguards for sigma: minimum and maximum step length.
        sigma_min = span / 1e6
        sigma_max = span * 2.0

        # Multiplicative factors for step‑size adaptation.
        factor_up = 1.1
        factor_down = 0.9

        # ------------------------------------------------------------------
        # Main optimisation loop.
        # ------------------------------------------------------------------
        while evals < self.budget:
            # Generate a random direction (isotropic).
            direction = np.random.randn(self.dim)
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue  # avoid division by zero (extremely unlikely)
            direction = direction / norm

            # Propose a new candidate point.
            candidate = x_current + sigma * direction
            # Enforce bound constraints.
            candidate = np.clip(candidate, lb, ub)

            # Evaluate the candidate.
            y_candidate = func(candidate)
            evals += 1

            # Selection: keep the better point.
            if y_candidate < best_y:
                # Accept improvement.
                x_current = candidate
                best_x = x_current.copy()
                best_y = y_candidate

                # Increase step size (exploration boost).
                sigma = np.clip(sigma * factor_up, sigma_min, sigma_max)
            else:
                # No improvement – shrink step size (exploitation focus).
                sigma = np.clip(sigma * factor_down, sigma_min, sigma_max)

        return best_x, best_y
