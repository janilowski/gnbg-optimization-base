# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1) Evolution Strategy (ES) with a 1/5‑success rule for step‑size adaptation.  
# The algorithm starts from a random feasible point, then iteratively generates a mutant by adding a Gaussian perturbation whose variance (step size) is continuously adjusted based on whether the mutant improves the objective.  
# Search state: Current best solution (x, y), current step size σ, and a counter of how many function evaluations have been performed.  
# Candidate generation: A single offspring is created by adding σ·N(0,1) to the current best vector.  
# Selection and replacement: If the offspring’s objective value is lower (better), it replaces the current best; otherwise the current best is retained.  
# Adaptation: σ is increased multiplicatively on success and decreased on failure, using an exponential update driven by the target success rate (≈20 %). This balances exploration (large σ) and exploitation (small σ).  
# Exploration mechanisms: Gaussian mutation provides isotropic exploration; σ is initially set to about one‑sixth of the feasible range, encouraging broad search early on.  
# Exploitation mechanisms: σ shrinks when no improvements are observed, focusing the search around the best point found so far.  
# Boundary handling: Candidate solutions are clipped to the provided lower/upper bounds (or a default ±10 if bounds are absent).  
# Budget strategy: One evaluation is spent on the initial random solution, then one evaluation per iteration. The loop stops as soon as the evaluation counter reaches the supplied budget, guaranteeing no budget overrun.  
# Closest known influences: Classical (1+1)-ES with the 1/5 rule from Rechenberg and Schwefel’s evolution strategies literature.  
# Novelty or unusual aspects: The implementation is intentionally minimal, using only NumPy, and adapts σ after every single evaluation to react quickly to changes in landscape shape.  
# Failure modes: The strategy can become trapped in local minima, especially on highly multi‑modal functions, but this is acceptable for a simple baseline on a limited budget.  
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple (1+1) Evolution Strategy with 1/5‑success rule.
    Minimises a black‑box function `func` while respecting the evaluation budget.
    """

    def __init__(self, budget: int, dim: int):
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
        Minimise `func` within the given evaluation budget.

        Parameters
        ----------
        func : callable
            Function that takes a NumPy array (shape = (dim,)) and returns a scalar.
            It is assumed to be a minimisation problem.

        Returns
        -------
        best_x : np.ndarray
            The best solution found (in the feasible region).
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------------
        # Determine the feasible region (bounds).
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = getattr(func.bounds, 'lb', None)
            upper = getattr(func.bounds, 'ub', None)
            if lower is None or upper is None:
                lower, upper = None, None
            else:
                lower = np.asarray(lower, dtype=float)
                upper = np.asarray(upper, dtype=float)
        else:
            lower, upper = None, None

        # If bounds are not provided, fall back to a default interval.
        if lower is None or upper is None:
            lower = np.full(self.dim, -10.0)
            upper = np.full(self.dim, 10.0)

        # Ensure the bounds vector length matches the dimensionality.
        if lower.shape[0] != self.dim:
            lower = np.resize(lower, self.dim)
        if upper.shape[0] != self.dim:
            upper = np.resize(upper, self.dim)

        # ------------------------------------------------------------------
        # Initial random solution.
        # ------------------------------------------------------------------
        x = np.random.random(self.dim)
        x = lower + x * (upper - lower)
        y = func(x)

        best_x = x.copy()
        best_y = float(y)
        evals = 1  # one evaluation performed

        # ------------------------------------------------------------------
        # Step‑size (σ) initialisation – roughly 1/6 of the search range.
        # ------------------------------------------------------------------
        sigma = np.mean(upper - lower) / 6.0

        # ------------------------------------------------------------------
        # Parameters for the 1/5‑success rule.
        # ------------------------------------------------------------------
        target_success = 0.2  # desired success rate
        success_count = 0      # successes observed

        # ------------------------------------------------------------------
        # Main optimisation loop.
        # ------------------------------------------------------------------
        while evals < self.budget:
            # Generate a candidate by isotropic Gaussian mutation.
            candidate = best_x + sigma * np.random.normal(size=self.dim)

            # Enforce bound constraints.
            candidate = np.clip(candidate, lower, upper)

            # Evaluate the candidate.
            cand_y = func(candidate)
            evals += 1

            # Selection: keep the better solution.
            if cand_y < best_y:
                best_x = candidate
                best_y = float(cand_y)
                success_count += 1
                # Increase step size (explore more).
                sigma *= np.exp(1.0 / (self.dim * target_success))
            else:
                # Decrease step size (exploit the current best).
                sigma *= np.exp(-1.0 / (self.dim * target_success))

            # Keep σ within a reasonable range relative to the domain.
            min_sigma = np.min(upper - lower) / 1e4
            max_sigma = np.max(upper - lower)
            sigma = min(max(sigma, min_sigma), max_sigma)

        return best_x, best_y
