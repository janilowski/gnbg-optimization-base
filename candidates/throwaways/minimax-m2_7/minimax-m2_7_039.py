# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A minimalist (1+1) Evolution Strategy with self-adaptive step size based on the 1/5th success rule. It maintains a single candidate solution, mutates it with isotropic Gaussian noise, respects bound constraints, and stops when the evaluation budget is exhausted.
# Search state: Current solution (parent), its objective value, step size sigma, evaluation counter, and the best solution found so far.
# Candidate generation: Offspring created by adding a normal‑distributed mutation vector scaled by sigma to the parent, then clipped to the problem’s bounds.
# Selection and replacement: Offspring replaces the parent if its objective value is no worse than the parent’s; otherwise the parent is retained.
# Adaptation: sigma is increased by a factor of 1.2 on successful mutations and decreased by 0.8 on failures, approximating the 1/5th rule. sigma is bounded to prevent overflow/underflow.
# Exploration mechanisms: Isotropic Gaussian mutation provides omnidirectional exploration; sigma controls the scale, allowing broad search when large and fine-grained search when small.
# Exploitation mechanisms: Accepting improvements focuses the search around promising regions; decreasing sigma after failures tightens the search locally.
# Boundary handling: Candidate solutions are clipped to the lower/upper bounds after mutation to ensure feasibility.
# Budget strategy: The algorithm performs evaluations until the budget is reached, counting each func call (including the initial parent evaluation) and halting immediately when the budget is exhausted.
# Closest known influences: Inspired by the simple (1+1) Evolution Strategy of Rechenberg and Schwefel; shares the self‑adaptation idea with CMA‑ES but without covariance matrix updates.
# Novelty or unusual aspects: Deliberately compact and readable, relying only on numpy and standard library; no recombination, archives, or advanced operators.
# Failure modes: May struggle when the budget is smaller than the dimensionality; isotropic mutations can be insufficient on highly anisotropic landscapes; simple step‑size adaptation may cause oscillations on certain functions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    A simple (1+1) Evolution Strategy for black‑box minimization.

    The algorithm maintains a single candidate solution and adapts its
    step size using the 1/5th success rule. It respects bound constraints
    and stops when the evaluation budget is exhausted.
    """

    def __init__(self, budget, dim):
        """
        Initialize the optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the problem.
        """
        self.budget = int(budget)
        self.dim = int(dim)
        if self.budget <= 0:
            raise ValueError("Budget must be a positive integer.")
        if self.dim <= 0:
            raise ValueError("Dimension must be a positive integer.")

    def __call__(self, func):
        """
        Run the optimization and return the best found solution.

        Parameters
        ----------
        func : callable
            Objective function to minimize. It must accept a 1‑D numpy
            array of length `dim` and return a scalar.

        Returns
        -------
        best_x : numpy.ndarray
            Best solution found (vector of length `dim`).
        best_y : float
            Objective value at `best_x`.
        """
        # ------------------------------------------------------------------
        # Detect bounds (support both attribute styles)
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot determine bounds: expected attributes "
                             "`lower`/`upper` or `bounds.lb`/`bounds.ub`.")

        # Broadcast scalars to full dimension if needed
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)

        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            raise ValueError("Bounds shape does not match the problem dimension.")

        # ------------------------------------------------------------------
        # Initialize state
        # ------------------------------------------------------------------
        # Initial step size: 10% of the average range per dimension
        range_vals = ub - lb
        sigma0 = float(np.mean(range_vals) * 0.1)
        if sigma0 <= 0:
            sigma0 = 1.0
        sigma = sigma0

        # Initial parent (uniform random within bounds)
        x = np.random.uniform(lb, ub)
        y = func(x)                     # first evaluation
        n_eval = 1

        best_x = x.copy()
        best_y = y

        # ------------------------------------------------------------------
        # Main loop: generate and evaluate offspring until budget exhausted
        # ------------------------------------------------------------------
        while n_eval < self.budget:
            # Generate candidate by mutating the parent
            candidate = x + sigma * np.random.normal(size=self.dim)
            # Clip to feasible region
            candidate = np.clip(candidate, lb, ub)

            # Evaluate candidate
            y_candidate = func(candidate)
            n_eval += 1

            # Update best solution if improvement found
            if y_candidate < best_y:
                best_x = candidate.copy()
                best_y = y_candidate

            # Selection and step‑size adaptation (1/5th rule approximation)
            if y_candidate <= y:
                # Accept offspring as new parent
                x = candidate
                y = y_candidate
                sigma *= 1.2   # increase step size on success
            else:
                sigma *= 0.8   # decrease step size on failure

            # Keep sigma within reasonable bounds to avoid stagnation
            sigma = max(sigma, 1e-8)
            sigma = min(sigma, float(np.max(range_vals)) * 10)

        return best_x, best_y
