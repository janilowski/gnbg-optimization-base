# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A minimalist (1+1)-Evolution Strategy with the 1/5 success rule for
#          black‑box minimization on the GNBG benchmark.
# Search state: Current best solution (parent) and mutation step‑size sigma.
# Candidate generation: Offspring = parent + sigma * N(0, I), clipped to bounds.
# Selection and replacement: Keep the better of parent and offspring.
# Adaptation: sigma is adjusted every 10*dim evaluations based on success rate:
#             increase by 1.2 if >20% successes, decrease by 0.8 otherwise.
# Exploration mechanisms: Large initial sigma (10% of bound range) for broad
#                        coverage; clipping guarantees feasible solutions.
# Exploitation mechanisms: Shrinking sigma when improvements become scarce
#                        focuses search near the current best.
# Boundary handling: Off‑range values are clipped to problem bounds; default
#                    bounds [-5,5] are used if bounds are not exposed.
# Budget strategy: Strictly counts each function call; loop stops when the
#                  allocated budget is exhausted.
# Closest known influences: Classical (1+1)-ES (Rechenberg, 1973) and the 1/5
#                          success rule common in Evolution Strategies.
# Novelty or unusual aspects: Pure NumPy implementation with no external
#                             dependencies beyond the standard library.
# Failure modes: May get trapped on highly rugged landscapes; however,
#                adaptive sigma provides a reasonable balance for many GNBG
#                instances.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple (1+1)-Evolution Strategy for GNBG benchmark minimization.

    The optimizer respects the supplied evaluation budget and adapts the
    mutation step‑size using the 1/5 success rule.
    """

    def __init__(self, budget: int, dim: int):
        """
        Initialize the optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimizer on the given objective function.

        Parameters
        ----------
        func : callable
            A black‑box objective function. May expose bounds via attributes
            `lower`/`upper` or `bounds.lb`/`bounds.ub`. If bounds are not
            provided, a default range of [-5, 5] for each dimension is used.

        Returns
        -------
        best_x : np.ndarray
            The best (lowest) solution found.
        best_y : float
            The objective value at `best_x`.
        """
        # Determine search bounds
        lb, ub = self._get_bounds(func)

        # Initial random solution (counts as one evaluation)
        parent_x = np.random.uniform(lb, ub, size=self.dim)
        parent_y = func(parent_x)
        best_x, best_y = parent_x.copy(), parent_y

        # If budget is one, we already have the best solution
        if self.budget <= 1:
            return best_x, best_y

        # Initial mutation step‑size (10% of the bound range)
        sigma = (ub - lb) * 0.1

        # Parameters for the 1/5 success rule
        block_size = 10 * self.dim
        successes = 0
        evals_in_block = 0

        evals = 1  # we already evaluated the parent

        # Main optimization loop
        while evals < self.budget:
            # Generate offspring
            offspring_x = parent_x + sigma * np.random.normal(size=self.dim)
            # Keep candidate within bounds
            np.clip(offspring_x, lb, ub, out=offspring_x)

            # Evaluate offspring
            offspring_y = func(offspring_x)
            evals += 1

            # Selection: keep the better individual
            if offspring_y < parent_y:
                parent_x = offspring_x
                parent_y = offspring_y
                successes += 1
                # Update global best if necessary
                if parent_y < best_y:
                    best_x = parent_x.copy()
                    best_y = parent_y

            evals_in_block += 1

            # Adapt sigma using the 1/5 rule after each block
            if evals_in_block >= block_size:
                success_rate = successes / evals_in_block
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                # Reset counters for next block
                successes = 0
                evals_in_block = 0

            # Keep sigma within numerically stable bounds
            sigma = np.clip(sigma, (ub - lb) * 1e-12, (ub - lb) * 10)

        return best_x, best_y

    def _get_bounds(self, func):
        """
        Extract lower and upper bounds from the function object.

        The function may expose bounds as `lower`/`upper` or `bounds.lb`/`bounds.ub`.
        If neither is present, default bounds of [-5, 5] are returned.

        Parameters
        ----------
        func : callable
            Objective function (or wrapper containing bounds).

        Returns
        -------
        lb, ub : np.ndarray
            Lower and upper bound arrays of shape (dim,).
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fall back to default bounds
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Ensure bounds are arrays even if they were scalars
        if lb.ndim == 0:
            lb = np.full(self.dim, float(lb))
        if ub.ndim == 0:
            ub = np.full(self.dim, float(ub))

        return lb, ub
