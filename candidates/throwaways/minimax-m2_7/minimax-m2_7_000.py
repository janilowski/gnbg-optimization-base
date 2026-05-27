# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1) evolution strategy that mutates the current best solution
#           with Gaussian noise and adapts the mutation strength sigma after each
#           evaluation based on immediate improvement.
# Search state: The best solution found so far (best_x, best_y) and the current
#               mutation strength sigma.
# Candidate generation: A single candidate is created by perturbing best_x with
#                       a Gaussian vector scaled by sigma.
# Selection and replacement: If the candidate improves the objective it replaces
#                             the current best; otherwise the best remains unchanged.
# Adaptation: sigma is multiplied by 1.2 on a successful move and by 0.8 otherwise,
#              keeping it within a safe range relative to the problem’s bounds.
# Exploration mechanisms: Large sigma encourages broad search; random Gaussian
#                         perturbations provide isotropic exploration.
# Exploitation mechanisms: Small sigma focuses search around the best known point.
# Boundary handling: Proposed points are clipped component‑wise to the problem’s
#                    lower and upper bounds.
# Budget strategy: Exactly `budget` function evaluations are performed; the loop
#                  terminates when the budget is exhausted.
# Closest known influences: Classical (1+1) Evolution Strategy with log‑normal
#                           mutation scaling, simplified to a single‑candidate
#                           iterative scheme.
# Novelty or unusual aspects: Immediate sigma adaptation after each evaluation,
#                             rather than after a batch of trials, provides a
#                             simple yet responsive balance between exploration
#                             and exploitation.
# Failure modes: May converge prematurely to a local minimum if sigma shrinks
#                too rapidly; also sensitive to the initial sigma choice.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple (1+1) evolution strategy for black‑box minimization.

    The algorithm starts by sampling a random point within the supplied bounds.
    Afterwards it repeatedly mutates the current best solution using Gaussian
    noise, adapts the mutation strength sigma after each evaluation, and keeps
    the better of the two points. The process stops after the given evaluation
    budget is exhausted.
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the minimization and return the best solution found.

        Parameters
        ----------
        func : callable
            A black‑box objective function that accepts a 1‑D NumPy array of
            length `dim` and returns a scalar to be minimized. The function
            must expose its bounds either via `func.lower` / `func.upper` or
            via `func.bounds.lb` / `func.bounds.ub`.

        Returns
        -------
        best_x : numpy.ndarray
            The coordinate of the best solution.
        best_y : float
            The objective value at ``best_x``.
        """
        # ------------------------------------------------------------------
        # 1. Retrieve problem bounds (lower and upper).
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot determine problem bounds: "
                             "expected func.lower/func.upper or func.bounds.lb/func.bounds.ub")

        # Ensure bounds are of correct shape (dim,)
        lb = np.atleast_1d(lb)
        ub = np.atleast_1d(ub)
        if lb.shape[0] != self.dim or ub.shape[0] != self.dim:
            raise ValueError("Dimension mismatch between bounds and specified dim")

        # ------------------------------------------------------------------
        # 2. Initial random sample.
        # ------------------------------------------------------------------
        best_x = np.random.uniform(lb, ub)
        best_y = func(best_x)
        evals = 1

        # ------------------------------------------------------------------
        # 3. Initialize mutation strength sigma based on the bound range.
        #    Using a fraction (1/6) of the mean range mimics the classic ES
        #    assumption of a normally distributed search space.
        # ------------------------------------------------------------------
        rng = ub - lb
        sigma = np.mean(rng) / 6.0
        sigma_min = 1e-8 * np.mean(rng) if np.mean(rng) > 0 else 1e-8
        sigma_max = np.mean(rng) / 2.0

        # ------------------------------------------------------------------
        # 4. Main evaluation loop: generate, clip, evaluate, adapt.
        # ------------------------------------------------------------------
        while evals < self.budget:
            # Generate a candidate by mutating the current best.
            candidate = best_x + sigma * np.random.randn(self.dim)
            # Enforce bound constraints component‑wise.
            np.clip(candidate, lb, ub, out=candidate)

            # Evaluate the candidate.
            cand_y = func(candidate)
            evals += 1

            # Selection: keep the better solution.
            if cand_y < best_y:
                best_x = candidate.copy()
                best_y = cand_y
                # Successful move -> increase exploration.
                sigma = min(sigma * 1.2, sigma_max)
            else:
                # No improvement -> reduce exploration radius.
                sigma = max(sigma * 0.8, sigma_min)

        return best_x, best_y
