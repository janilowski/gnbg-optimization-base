# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: The algorithm follows a two‑phase scheme. First it performs a modest
# random sweep of the domain to locate promising regions. Afterwards it switches
# to a (1+1) Evolution Strategy with a self‑adaptive step size (sigma) that
# performs a local hill‑climbing search.  This yields a balance between broad
# exploration and focused exploitation while staying within the supplied
# evaluation budget.
#
# Search state: The optimizer keeps the current point (x), the best point found
# (best_x), the current function value (y), the best function value (best_y),
# and an array of step sizes (sigma) – one per dimension.  A counter tracks how
# many function evaluations have been performed.
#
# Candidate generation: A new candidate is created by adding Gaussian noise to
# the current point: candidate = x + sigma * N(0, I).  The noise is generated
# with numpy's standard normal generator, which the calling harness seeds
# before each run.
#
# Selection and replacement: The candidate replaces the current point only if it
# yields a lower (or equal) objective value.  The global best is updated
# whenever the current point improves.
#
# Adaptation: sigma is multiplied by 1.1 after a successful step and by 0.9 after
# a failure.  The factor 1.1/0.9 implements a simple 1/5‑success rule.  sigma
# is clipped to stay within a tiny fraction (1e‑6) and half of the original
# variable range, preventing it from collapsing to zero or exploding.
#
# Exploration mechanisms: The initial random sampling (≈20 % of the budget)
# provides a coarse exploration of the whole domain.  A successful step enlarges
# sigma, encouraging larger moves and thus broader exploration.
#
# Exploitation mechanisms: When the candidate does not improve, sigma shrinks,
# focusing the search around the current point – a classic hill‑climbing
# behavior.
#
# Boundary handling: After generating a candidate, each component is clipped to
# the interval [lower, upper] (or [lb, ub] depending on the attribute names).
# This keeps the search inside the admissible region.
#
# Budget strategy: The algorithm counts every call to the objective function.
# The main loop terminates as soon as the evaluation counter reaches the
# supplied budget, guaranteeing that the budget is never exceeded.
#
# Closest known influences: Random sampling combined with a (1+1) Evolution
# Strategy and a 1/5‑success step‑size adaptation.  Similar ideas appear in
# simple evolutionary algorithms and in the classic Bump–Human algorithm.
#
# Novelty or unusual aspects: The fixed 20 % random‑sampling phase before the
# ES is a deliberate design choice to ensure a decent starting point for the
# local search, even when the budget is modest.
#
# Failure modes: If the objective contains many sharp local minima, sigma may
# shrink prematurely, causing the search to get stuck.  The random‑sampling
# phase mitigates this to some extent, but very deceptive landscapes can still
# degrade performance.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Black‑box minimizer using a short random sampling phase followed by a
    (1+1) Evolution Strategy with a self‑adaptive step size.
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the decision space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Minimise `func` within the allowed evaluation budget.

        Parameters
        ----------
        func : callable
            Objective function.  Expected to accept a 1‑D numpy array of
            length `dim` and return a scalar.

        Returns
        -------
        best_x : numpy.ndarray
            Decision vector that achieved the smallest observed value.
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------------
        # Determine variable bounds (prefer attributes lower/upper; fallback
        # to bounds.lb/ub; finally default to [0,1] for all dimensions).
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            lb = np.zeros(self.dim, dtype=float)
            ub = np.ones(self.dim, dtype=float)

        # Ensure lb/ub are arrays of the correct shape.
        lb = np.atleast_1d(lb)
        ub = np.atleast_1d(ub)
        if lb.shape[0] != self.dim:
            lb = np.full(self.dim, lb[0])
        if ub.shape[0] != self.dim:
            ub = np.full(self.dim, ub[0])

        # Search range for each dimension.
        range_ = ub - lb

        # ------------------------------------------------------------------
        # Phase 1 – random sampling (≈20 % of the budget, at least one point).
        # ------------------------------------------------------------------
        sample_size = max(1, int(np.ceil(0.2 * self.budget)))
        evals = 0
        best_x = None
        best_y = np.inf

        for _ in range(sample_size):
            x = lb + range_ * np.random.rand(self.dim)
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()

        # Current point for the local search starts from the best sampled point.
        cur_x = best_x.copy()
        cur_y = best_y

        # ------------------------------------------------------------------
        # Phase 2 – (1+1) Evolution Strategy with self‑adaptive sigma.
        # ------------------------------------------------------------------
        # Initial step size: 1/10 of the variable range (per dimension).
        sigma = range_ / 10.0
        # Bounds on sigma to prevent it from vanishing or exploding.
        sigma_min = 1e-6 * range_
        sigma_max = range_ / 2.0

        while evals < self.budget:
            # Generate a candidate by adding Gaussian noise scaled by sigma.
            candidate = cur_x + sigma * np.random.randn(self.dim)
            # Keep the candidate inside the feasible region.
            candidate = np.clip(candidate, lb, ub)

            # Evaluate the candidate.
            cand_y = func(candidate)
            evals += 1

            # Selection: accept only if the candidate improves the current value.
            if cand_y <= cur_y:
                cur_x = candidate
                cur_y = cand_y
                # Successful step → increase sigma (explore more).
                sigma = np.clip(sigma * 1.1, sigma_min, sigma_max)
                # Update global best if needed.
                if cur_y < best_y:
                    best_y = cur_y
                    best_x = cur_x.copy()
            else:
                # Unsuccessful step → shrink sigma (exploit).
                sigma = np.clip(sigma * 0.9, sigma_min, sigma_max)

            # If we have exhausted the budget, exit the loop.
            if evals >= self.budget:
                break

        return best_x, best_y
