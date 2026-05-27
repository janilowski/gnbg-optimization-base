# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a simple, compact evolutionary strategy for
#          black‑box minimization. It starts with a small random sample to locate
#          a reasonable region, then refines the best solution using a (1+1)-ES
#          with step‑size (sigma) adaptation following the 1/5‑rule.
# Search state: The optimizer maintains the current best solution (x_best,
#                y_best) and the current step size sigma.
# Candidate generation: A candidate is produced by adding a Gaussian perturbation
#                       (sigma * N(0,I)) to the current best solution.
# Selection and replacement: If the candidate's objective value is less than or
#                           equal to the best known value, it replaces the best
#                           point; otherwise the best point remains unchanged.
# Adaptation: Sigma is increased by a factor (1.1) after a successful step and
#             decreased by a factor (0.9) after a failure, keeping it within
#             reasonable bounds. This implements a simple 1/5‑rule.
# Exploration mechanisms: Initial random sampling (up to 10*dim points) ensures
#                         a broad coverage of the search space; sigma can grow
#                         to explore larger regions when improvements occur.
# Exploitation mechanisms: Sigma shrinks when no improvement is observed,
#                          focusing the search around the current best.
# Boundary handling: Candidates are clipped to the provided lower/upper bounds
#                     to stay within the feasible region.
# Budget strategy: The evaluation budget is respected exactly: a small fraction
#                  of the budget is used for the initial random sample, the
#                  remainder is allocated to the (1+1)-ES iterations.
# Closest known influences: Classic (1+1)-ES, CMA‑ES ideas (step‑size adaptation),
#                          and simple random search with local refinement.
# Novelty or unusual aspects: The implementation is deliberately minimalistic,
#                            relying only on numpy and standard library features,
#                            making it easy to embed in any benchmarking harness.
# Failure modes: The algorithm may get stuck in local minima on highly
#                multi‑modal landscapes; sigma can become extremely small
#                after many failures, leading to stagnation.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple black‑box minimizer using a (1+1)-ES with random initial sampling.

    Public interface:
        __init__(self, budget, dim)  – store budget and problem dimension
        __call__(self, func)         – minimize func, return (best_x, best_y)
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        """
        Run the optimization within the evaluation budget.

        Parameters
        ----------
        func : callable
            Black‑box objective. Must accept a 1‑D numpy array of length dim
            and return a scalar. Bounds are read from func.lower / func.upper
            or func.bounds.lb / func.bounds.ub.

        Returns
        -------
        best_x : numpy.ndarray
            Best solution found (1‑D array of length dim).
        best_y : float
            Objective value at best_x.
        """
        # ------------------------------------------------------------------
        # Obtain problem bounds (lower and upper arrays of length dim)
        # ------------------------------------------------------------------
        lb, ub = self._get_bounds(func)

        # ------------------------------------------------------------------
        # Guard against zero or negative budget – return a random feasible
        # point (the harness should normally provide a positive budget).
        # ------------------------------------------------------------------
        if self.budget <= 0:
            x = self._random_point(lb, ub)
            y = func(x)
            return x, y

        # ------------------------------------------------------------------
        # 1) Initial random sampling to obtain a decent starting point.
        #    We allocate a modest fraction of the budget (up to 10*dim or
        #    budget‑1 points) for this phase.
        # ------------------------------------------------------------------
        # First evaluation – always performed to have a valid best point.
        best_x = self._random_point(lb, ub)
        best_y = func(best_x)
        evals = 1

        # Additional random points (if budget allows)
        max_extra = min(self.budget - evals, max(1, 10 * self.dim))
        for _ in range(max_extra):
            x = self._random_point(lb, ub)
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x

        # ------------------------------------------------------------------
        # 2) (1+1)-ES with step‑size adaptation for the remaining budget.
        # ------------------------------------------------------------------
        remaining = self.budget - evals
        if remaining > 0:
            # Initial step size – a fraction of the domain span.
            span = ub - lb
            sigma = np.median(span) / 5.0
            sigma = max(sigma, 1e-12)          # avoid zero
            sigma_max = np.max(span) / 2.0
            factor_up = 1.1
            factor_down = 0.9

            for _ in range(remaining):
                # Generate a candidate by Gaussian perturbation.
                cand = best_x + sigma * np.random.randn(self.dim)
                # Clip to stay inside the feasible region.
                cand = np.clip(cand, lb, ub)

                y_cand = func(cand)
                evals += 1

                # Accept or reject based on improvement.
                if y_cand <= best_y:
                    best_x = cand
                    best_y = y_cand
                    sigma = min(sigma * factor_up, sigma_max)
                else:
                    sigma = max(sigma * factor_down, 1e-12)

        return best_x, best_y

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def _random_point(self, lb, ub):
        """Return a uniformly random point inside the box defined by lb, ub."""
        return np.random.uniform(lb, ub)

    def _get_bounds(self, func):
        """
        Extract lower and upper bounds from the function object.

        Expected attributes:
            func.lower / func.upper   – arrays (or array‑like) of length dim
        or
            func.bounds.lb / func.bounds.ub   – same.

        Returns
        -------
        lb, ub : numpy.ndarray
            Arrays of shape (dim,).
        """
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb, dtype=float)
                ub = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError(
                    "Cannot determine bounds: func must have either "
                    "lower/upper or bounds.lb/bounds.ub attributes."
                )

        # Ensure they are 1‑D arrays of length dim.
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)

        if lb.shape[0] != self.dim or ub.shape[0] != self.dim:
            raise ValueError(
                f"Bounds dimension mismatch: expected {self.dim}, "
                f"got lower={lb.shape[0]}, upper={ub.shape[0]}."
            )
        return lb, ub
