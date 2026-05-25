import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implementation of CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box minimization. Uses a population of candidate solutions, adapts the covariance matrix and step size to efficiently explore the search space.
# Search state: Mean vector (m), covariance matrix (C), step-size (sigma), evolution paths (p_c, p_s), best solution found so far.
# Candidate generation: Samples lambda candidates from a multivariate normal distribution: m + sigma * N(0, C). Then applies boundary reflection to enforce box constraints.
# Selection and replacement: After evaluating all candidates, selects the best mu candidates based on fitness (minimization). Updates the mean as weighted sum of selected points (using recombination weights).
# Adaptation: Updates evolution paths using differential evolution of the mean. Covariance matrix adaptation uses rank-one and rank-mu updates. Step-size adaptation uses cumulative step-length control (CSA).
# Exploration mechanisms: Early steps are exploratory due to large step-size and isotropic covariance. The adaptation allows changes in preferred directions.
# Exploitation mechanisms: As covariance shrinks along promising directions, sampling concentrates near good solutions. The mean moves towards better regions.
# Boundary handling: Reflection: if a coordinate exceeds [lower, upper], it is reflected back into the bounds. This preserves diversity and avoids clustering at boundaries.
# Budget strategy: Uses the full budget. Stops when evaluations used >= budget. No early termination.
# Closest known influences: Standard CMA-ES (Hansen, 2016). Implementation follows the pseudo-code with minor simplifications (e.g., fixed population size, no restart).
# Novelty or unusual aspects: Compact implementation using only numpy. Boundary handling via reflection is not in all CMA-ES variants but commonly used.
# Failure modes: May converge prematurely if step-size shrinks too fast; may struggle in rugged landscapes. Performance degrades in very high dimensions due to population size scaling. Not suitable for discrete or mixed-variable optimization.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        """Initialize CMA-ES with given budget (function evaluations) and dimension."""
        self.budget = int(budget)
        self.dim = int(dim)
        # Population size (lambda) – heuristic from Hansen's CMA-ES
        self.lam = max(2, 4 + int(3 * np.log(self.dim)))
        # Number of selected parents (mu)
        self.mu = self.lam // 2
        # Recombination weights (logarithmic decreasing)
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)  # variance effective selection mass

        # Strategy parameter setting (defaults from CMA-ES)
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.dim + 2.0) ** 2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / (self.dim + 1.0)) - 1.0) + self.cs

        # Internal state (will be initialized in __call__)
        self.best_x = None
        self.best_y = np.inf

    def _reflect(self, x, lb, ub):
        """Reflect coordinates that are outside bounds back into the domain."""
        for i in range(self.dim):
            if x[i] < lb[i]:
                x[i] = lb[i] + (lb[i] - x[i])
            elif x[i] > ub[i]:
                x[i] = ub[i] - (x[i] - ub[i])
        return x

    def __call__(self, func):
        """Run CMA-ES minimization. Returns (best_x, best_y)."""
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via .lower/.upper or .bounds.lb/.bounds.ub")

        # Initialize state
        # Mean: uniform random in the domain
        m = lb + np.random.rand(self.dim) * (ub - lb)
        sigma = 0.5 * np.max(ub - lb)  # initial step size (half of domain range)
        C = np.eye(self.dim)
        p_c = np.zeros(self.dim)
        p_s = np.zeros(self.dim)

        # Evaluate initial mean (use one budget call)
        evals = 0
        y = float(func(m))
        evals += 1
        if y < self.best_y:
            self.best_y = y
            self.best_x = m.copy()

        # Precompute constant for chi_n
        chi_n = np.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))

        # Main optimization loop
        while evals < self.budget:
            # Generate and evaluate lambda offspring
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: add small diagonal to ensure positive definiteness
                C += 1e-9 * np.eye(self.dim)
                L = np.linalg.cholesky(C)

            # Preallocate arrays
            ar_x = np.empty((self.lam, self.dim))
            ar_y = np.empty(self.lam)

            for i in range(self.lam):
                # Sample from N(0, C)
                z = np.random.randn(self.dim)
                x = m + sigma * (L @ z)
                # Reflect boundaries
                x = self._reflect(x, lb, ub)
                ar_x[i] = x
                ar_y[i] = float(func(x))
                evals += 1
                if evals > self.budget:
                    break

            # Check budget after generation (may have exactly reached budget)
            if evals >= self.budget:
                break

            # Update best solution
            idx_min = np.argmin(ar_y)
            if ar_y[idx_min] < self.best_y:
                self.best_y = ar_y[idx_min]
                self.best_x = ar_x[idx_min].copy()

            # Sort by fitness (minimization)
            sorted_idx = np.argsort(ar_y)
            ar_x_sorted = ar_x[sorted_idx]
            ar_y_sorted = ar_y[sorted_idx]

            # Select mu best
            x_sel = ar_x_sorted[:self.mu]

            # Update mean
            m_old = m.copy()
            m = np.dot(self.weights, x_sel)

            # Update evolution paths
            delta = m - m_old
            # p_c update
            p_c = (1.0 - self.cc) * p_c + np.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * delta / sigma

            # p_s update
            # Need C^(-1/2) * delta; compute using eigendecomposition for stability, but we can use Cholesky inverse
            try:
                C_inv = np.linalg.inv(C)
                # sqrt of C: use eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                # C^(1/2) = U * diag(sqrt(eigvals)) * U^T
                sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                # inv(sqrt_C) = U * diag(1/sqrt(eigvals)) * U^T
                inv_sqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                # C^(-1/2) * (m - m_old) / sigma
                delta_scaled = inv_sqrt_C @ (delta / sigma)
            except np.linalg.LinAlgError:
                # Fallback: use identity for pathological cases
                delta_scaled = delta / sigma

            p_s = (1.0 - self.cs) * p_s + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * delta_scaled

            # Update covariance matrix (rank-one + rank-mu)
            # Rank-one update
            C = (1.0 - self.c1 - self.cmu) * C \
                + self.c1 * (np.outer(p_c, p_c)) \
                + self.cmu * sum(
                    self.weights[i] * np.outer((x_sel[i] - m_old) / sigma, (x_sel[i] - m_old) / sigma)
                    for i in range(self.mu)
                )

            # Update step-size
            sigma = sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(p_s) / chi_n - 1.0))

            # Enforce positivity of sigma
            sigma = max(sigma, 1e-10)

        return (self.best_x, self.best_y)
