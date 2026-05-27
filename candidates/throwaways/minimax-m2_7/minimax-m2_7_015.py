# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer that minimizes a black‑box function within a fixed evaluation budget. It maintains a multivariate normal distribution over the search space, adapts its mean, step‑size (σ) and covariance matrix using the classic CMA-ES update rules, and respects problem bounds by clipping candidate points.
# Search state: The algorithm holds a distribution mean (dim‑dimensional vector), a step‑size σ (scalar), a covariance matrix C (dim×dim), and two evolution paths ps and pc (both dim‑dimensional). Counters track the number of evaluations performed and the current iteration.
# Candidate generation: At each iteration a batch of λ points is drawn from N(mean, σ²·C). Points outside the problem bounds are clipped to the nearest bound (simple projection). The batch size λ is adapted to the budget when fewer evaluations remain.
# Selection and replacement: After evaluating the batch, the µ best points (according to objective value) are retained. The new mean is recomputed as a weighted sum of the selected points, using positive weights that favor better candidates.
# Adaptation: The step‑size σ is updated using the cumulative evolution path ps and the so‑called sigma increase‑rate (CSA). The covariance matrix C is updated with a rank‑1 term (based on pc) and a rank‑µ term (based on the selected candidate deviations). Parameters follow the standard CMA-ES tuning formulas that depend on µ_eff and dimension.
# Exploration mechanisms: Large σ promotes exploration; the covariance matrix adaptation gradually structures the search to follow the landscape’s curvature.
# Exploitation mechanisms: As σ shrinks, candidate points concentrate around the current best region; the weighted selection pushes the mean toward promising solutions.
# Boundary handling: Candidate vectors are clipped to the problem’s lower/upper limits. This is a simple, budget‑preserving projection that prevents evaluation of out‑of‑bounds points.
# Budget strategy: The algorithm never exceeds the provided evaluation budget. When the remaining budget is smaller than λ, it samples a reduced batch. After each evaluation the internal counter is checked and the main loop exits once the budget is exhausted.
# Closest known influences: The implementation follows the classic CMA-ES described by Hansen & Ostermeier (2001) and the compact parameter settings recommended in the literature.
# Novelty or unusual aspects: The optimizer is self‑contained (only NumPy and standard library), does not rely on external optimization libraries, and includes a fallback default bound when the function does not expose bounds.
# Failure modes: With very low budgets the algorithm may not converge; CMA‑ES can struggle on highly rugged or heavily constrained landscapes, but it remains a robust general‑purpose method for continuous black‑box optimization.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    CMA-ES based optimizer for black‑box minimization.

    Public interface:
        algo = Algorithm(budget, dim)
        best_x, best_y = algo(objective_func)
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective evaluations allowed.
        dim : int
            Dimensionality of the decision space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimizer on the given objective function.

        Parameters
        ----------
        func : callable
            A black‑box objective: func(x) -> float, where x is a 1‑D array
            of length dim. The function may expose bounds via:
                - func.lower / func.upper, or
                - func.bounds.lb / func.bounds.ub.
            If neither is present, default bounds [-5,5]^dim are used.

        Returns
        -------
        best_x : np.ndarray
            Decision vector that achieved the lowest observed objective value.
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------------
        # 1. Determine problem bounds
        # ------------------------------------------------------------------
        lower, upper = self._read_bounds(func)

        # ------------------------------------------------------------------
        # 2. Initialize CMA-ES state
        # ------------------------------------------------------------------
        # Mean of the search distribution
        mean = 0.5 * (lower + upper)

        # Initial step‑size (heuristic: about one third of the search range)
        sigma = np.mean(upper - lower) / 3.0

        # Identity covariance matrix
        C = np.eye(self.dim)

        # Evolution paths
        ps = np.zeros(self.dim)   # for sigma adaptation
        pc = np.zeros(self.dim)   # for covariance adaptation

        # ------------------------------------------------------------------
        # 3. CMA-ES hyper‑parameters (standard formulas)
        # ------------------------------------------------------------------
        # Population size λ and number of parents µ
        lambda_ = int(4 + 3 * np.log(self.dim))
        mu = int(lambda_ // 2)

        # Positive weights for recombination of the µ best candidates
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()

        # Variance‑effective mass of the selected distribution
        mu_eff = (weights.sum() ** 2) / (weights ** 2).sum()

        # Adaptation constants
        cc = 4.0 / (self.dim + 4.0)
        cs = (mu_eff + 2.0) / (self.dim + mu_eff + 3.0)
        c1 = 2.0 / ((self.dim + 1.3) ** 2 + mu_eff)
        cmu = min(1.0 - c1, 2.0 * (mu_eff - 1.0 + 1.0 / mu_eff) /
                  ((self.dim + 2.0) ** 2 + mu_eff))
        dampscale = 1.0 + max(0.0, np.sqrt(mu_eff / self.dim) - 1.0)

        # Expectation of ||N(0,I)|| (norm of a standard normal)
        chi_n = np.sqrt(self.dim) * (1.0 + 1.0 / (4.0 * self.dim) +
                                      1.0 / (21.0 * self.dim ** 2))

        # ------------------------------------------------------------------
        # 4. Initial evaluation of the mean point (counts toward budget)
        # ------------------------------------------------------------------
        best_x = mean.copy()
        best_y = func(best_x)
        evals = 1

        if evals >= self.budget:
            return best_x, best_y

        # ------------------------------------------------------------------
        # 5. Main CMA-ES loop
        # ------------------------------------------------------------------
        while evals < self.budget:
            # Determine how many candidates we can draw in this iteration
            remaining = self.budget - evals
            sample_n = lambda_ if remaining >= lambda_ else remaining

            # ------------------------------------------------------------------
            # 5a. Sample λ (or fewer) candidates from N(mean, σ²·C)
            # ------------------------------------------------------------------
            # Draw random vectors from the multivariate normal
            z = np.random.multivariate_normal(np.zeros(self.dim), C, size=sample_n)
            X = mean + sigma * z                     # shape (sample_n, dim)

            # Project out‑of‑bounds points onto the feasible hyper‑rectangle
            X = np.clip(X, lower, upper)

            # ------------------------------------------------------------------
            # 5b. Evaluate candidates
            # ------------------------------------------------------------------
            y = np.empty(sample_n)
            for i in range(sample_n):
                y[i] = func(X[i])
                evals += 1

            # Track current best
            best_idx = np.argmin(y)
            if y[best_idx] < best_y:
                best_y = y[best_idx]
                best_x = X[best_idx].copy()

            # If budget exhausted, stop (do not perform adaptation)
            if evals >= self.budget:
                break

            # ------------------------------------------------------------------
            # 5c. Selection – keep the µ best individuals
            # ------------------------------------------------------------------
            sorted_idx = np.argsort(y)          # ascending (minimisation)
            selected_idx = sorted_idx[:mu]       # µ best indices
            X_sel = X[selected_idx]              # shape (µ, dim)

            # ------------------------------------------------------------------
            # 5d. Recombination – update mean
            # ------------------------------------------------------------------
            mean_old = mean.copy()
            mean = np.dot(weights, X_sel)        # shape (dim,)

            # ------------------------------------------------------------------
            # 5e. Compute normalized differences y_i = (X_i - mean_old) / σ
            # ------------------------------------------------------------------
            Y = (X_sel - mean_old) / sigma       # shape (µ, dim)

            # ------------------------------------------------------------------
            # 5f. Update evolution paths ps (sigma) and pc (covariance)
            # ------------------------------------------------------------------
            # Sigma path update
            delta_mean = (mean - mean_old) / sigma
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mu_eff) * delta_mean

            # Sigma step‑size adaptation (CSA)
            sigma_mult = np.exp((cs / dampscale) * (np.linalg.norm(ps) / chi_n - 1.0))
            sigma *= sigma_mult

            # Covariance path update
            pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc) * mu_eff) * delta_mean

            # ------------------------------------------------------------------
            # 5g. Update covariance matrix C (rank‑1 + rank‑µ updates)
            # ------------------------------------------------------------------
            # Rank‑1 term
            C = (1.0 - c1 - cmu) * C + c1 * np.outer(pc, pc)

            # Rank‑µ term
            # weights[:, None] multiplies each row of Y
            C += cmu * ((weights[:, None] * Y).T @ Y)

            # Ensure symmetry (numerical safety)
            C = (C + C.T) / 2.0

            # (Optional) keep C positive‑definite by clipping negative eigenvalues.
            # For brevity we omit this step – standard CMA‑ES rarely violates it.

        # ------------------------------------------------------------------
        # 6. Return best found solution
        # ------------------------------------------------------------------
        return best_x, best_y

    def _read_bounds(self, func):
        """
        Extract lower and upper bounds from the function object.

        The function may expose bounds in one of the following ways:
            - func.lower  / func.upper  (attributes)
            - func.bounds.lb / func.bounds.ub (attributes)
        If neither is present, default bounds [-5, 5]^dim are used.

        Returns
        -------
        lower, upper : np.ndarray
            Column vectors of shape (dim,) representing the box constraints.
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback default: hyper‑cube [-5,5]^dim
            lower = np.full(self.dim, -5.0)
            upper = np.full(self.dim, 5.0)
        return lower, upper
