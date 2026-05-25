import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (1+1)-Evolution Strategy with cumulative step‑size adaptation based on exponential smoothing of success probability. It generates isotropic Gaussian mutations around the current best solution and adapts the global step size to maintain a success rate near 0.2.
# Search state: best_x (current elite), best_y (its objective value), sigma (global step size), p_success (smoothed success probability).
# Candidate generation: best_x + sigma * np.random.randn(dim).
# Selection and replacement: If candidate objective < best_y, replace best; otherwise keep the current best.
# Adaptation: Exponential moving average of success: p_success = (1-c)*p_success + c*success_flag; then sigma *= exp((p_success-0.2)/d). Parameters: c = 0.1, d = 0.5.
# Exploration mechanisms: Isotropic Gaussian mutations with adaptive step size.
# Exploitation mechanisms: Greedy elitist selection; step size shrinks when success rate is too low, increasing exploitation around the current best.
# Boundary handling: Clipping candidate coordinates to the provided lower and upper bounds.
# Budget strategy: Uses the entire evaluation budget; stops exactly when the budget is exhausted.
# Closest known influences: Classic (1+1)-ES with 1/5th rule, here implemented via exponential smoothing instead of a fixed window.
# Novelty or unusual aspects: None – this is a well‑known, compact evolutionary strategy.
# Failure modes: Greedy selection may cause premature convergence to local minima; isotropic Gaussian is inefficient for highly anisotropic or ill‑conditioned landscapes; no restart mechanism; step size may collapse in flat regions.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
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
        Run the optimization on a given objective function.

        Parameters
        ----------
        func : callable
            Objective function with attributes lower/upper or bounds.lb/bounds.ub
            providing the search domain.

        Returns
        -------
        best_x : ndarray of shape (dim,)
            Best solution found.
        best_y : float
            Objective value of the best solution.
        """
        # --- read bounds ---------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            bounds = func.bounds
            lower = np.asarray(bounds.lb, dtype=float)
            upper = np.asarray(bounds.ub, dtype=float)
        else:
            raise ValueError("Function must provide lower/upper or bounds.lb/bounds.ub")
        dim = self.dim
        lower = lower.ravel()
        upper = upper.ravel()

        # --- initialisation ------------------------------------------------
        best_x = lower + (upper - lower) * np.random.rand(dim)
        best_y = func(best_x)
        evaluations = 1

        # initial step size ~ 1/4 of average range
        sigma = 0.25 * (upper - lower).mean()

        # smoothed success probability (initialised slightly above target)
        p_success = 0.5
        # adaptation parameters
        c = 0.1          # smoothing factor
        d = 0.5          # damping factor
        target = 0.2     # desired success probability

        # --- main loop -----------------------------------------------------
        while evaluations < self.budget:
            # generate candidate
            candidate = best_x + sigma * np.random.randn(dim)
            # clip to bounds
            candidate = np.clip(candidate, lower, upper)
            # evaluate
            candidate_y = func(candidate)
            evaluations += 1

            # selection
            improvement = candidate_y < best_y
            if improvement:
                best_x, best_y = candidate, candidate_y

            # update success probability (exponential smoothing)
            p_success = (1.0 - c) * p_success + c * float(improvement)

            # adjust step size
            sigma *= np.exp((p_success - target) / d)
            # prevent step size from becoming too small or too large
            sigma = np.clip(sigma, 1e-12 * (upper - lower).max(), 0.5 * (upper - lower).max())

        return best_x, best_y
