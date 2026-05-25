# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for continuous black-box minimization.
#   Uses rank-one and rank-mu updates, cumulative step-size adaptation (CSA), and a variable population size.
# Search state: mean vector (m), covariance matrix (C), step size (sigma), evolution paths (p_c, p_s).
# Candidate generation: lambda offspring sampled from N(m, sigma^2 * C) via Cholesky decomposition.
# Selection and replacement: Truncation selection — top mu offspring are used to update the mean and covariance.
# Adaptation: Step-size updated via cumulative step-size adaptation; covariance updated via rank-one and rank-mu rules.
# Exploration mechanisms: Initial large sigma (0.5 * domain range) and random sampling from multivariate normal.
# Exploitation mechanisms: Covariance matrix learns the underlying Hessian, step size adapts to encourage convergence.
# Boundary handling: Sampled offspring are clipped to the variable bounds (lower/upper).
# Budget strategy: Stop after exactly budget function evaluations have been used; if budget is smaller than one generation, fall back to pure random search.
# Closest known influences: Standard CMA-ES as introduced by Hansen (2016).
# Novelty or unusual aspects: None; straight‑forward implementation for robustness across dimensions.
# Failure modes: May converge prematurely on highly multi‑modal functions; poor performance if budget is extremely small relative to dimension.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """CMA-ES minimizer for black‑box functions under an evaluation budget."""

    def __init__(self, budget: int, dim: int) -> None:
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ------------------------------------------------------------------
        # Read domain bounds from func (supports both common patterns)
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=np.float64)
            ub = np.asarray(func.upper, dtype=np.float64)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=np.float64)
            ub = np.asarray(func.bounds.ub, dtype=np.float64)
        else:
            # Fallback: assume symmetric bounds around zero; should not happen
            lb = np.full(self.dim, -1e6)
            ub = np.full(self.dim,  1e6)

        # Ensure correct shape
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)

        domain_range = ub - lb
        if np.any(domain_range <= 0):
            raise ValueError("Lower bound must be strictly smaller than upper bound.")

        # ------------------------------------------------------------------
        # CMA-ES parameters (computed from budget and dimension)
        # ------------------------------------------------------------------
        n = self.dim
        # Population size – standard choice (saturate to budget)
        lam = max(2, min(self.budget, 4 + int(3 * np.log(n))))
        mu = lam // 2

        # Recombination weights and effective mu
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates (from Hansen 2016)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)          # cumulation for covariance
        cs = (mueff + 2) / (n + mueff + 5)                       # cumulation for step‑size
        c1 = 2 / ((n + 1.3) ** 2 + mueff)                       # rank‑one learning rate
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))  # rank‑mu learning rate
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs  # damping for step‑size

        # Initial mean (center of domain) and step‑size (25% of range)
        m = 0.5 * (lb + ub)
        sigma = 0.25 * np.mean(domain_range)

        # Evolution paths and covariance matrix
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)

        # For Cholesky decomposition
        eig_values = None      # not used in plain Cholesky version
        B = None

        # ------------------------------------------------------------------
        # Helper: sample lambda points inside bounds
        # ------------------------------------------------------------------
        def sample_offspring(mean, cov, step_size, n_offspring):
            """Return (lambda_offspring, (n, lambda)) matrix."""
            # Cholesky factor L such that C = L L^T
            L = np.linalg.cholesky(cov)
            # Standard normal samples (n, lambda)
            z = np.random.randn(n, n_offspring)
            # Affine transformation: mean + sigma * L * z
            points = mean[:, np.newaxis] + step_size * (L @ z)
            # Clip to bounds
            points = np.clip(points, lb[:, np.newaxis], ub[:, np.newaxis])
            return points

        # ------------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------------
        evals = 0
        best_x = None
        best_y = np.inf

        # If the budget is smaller than a single generation, do pure random search
        if self.budget < lam:
            n_rand = self.budget
            points = np.random.uniform(lb[:, np.newaxis], ub[:, np.newaxis], (n, n_rand))
            for k in range(n_rand):
                x = points[:, k]
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # CMA-ES generations
        while evals + lam <= self.budget:
            # Sample lambda points
            points = sample_offspring(m, C, sigma, lam)

            # Evaluate all offspring
            y_vals = np.empty(lam)
            for k in range(lam):
                y_vals[k] = func(points[:, k])
                evals += 1

            # Update best seen
            idx_best = np.argmin(y_vals)
            if y_vals[idx_best] < best_y:
                best_y = y_vals[idx_best]
                best_x = points[:, idx_best].copy()

            # Sort by fitness
            idx_sorted = np.argsort(y_vals)
            y_vals_sorted = y_vals[idx_sorted]
            points_sorted = points[:, idx_sorted]

            # Compute weighted mean of the mu best points
            old_m = m.copy()
            m = (points_sorted[:, :mu] @ weights)   # (n,)

            # Compute step‑size adaptation
            #   c_sigma update (rank-one)
            c = np.sqrt(cc * (2 - cc) * mueff)
            pc = (1 - cc) * pc + c * (m - old_m) / sigma

            #   d_sigma update
            d = np.sqrt(cs * (2 - cs) * mueff)
            ps = (1 - cs) * ps + d * np.linalg.solve(C, (m - old_m) / sigma)

            # Update step size
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(cs / damps * (norm_ps / (np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))) - 1))

            # Update covariance matrix
            # Rank-one update
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)   # rank-one

            # Rank-mu update
            # Compute weighted outer product of the mu best deviations
            dev = (points_sorted[:, :mu] - old_m[:, np.newaxis]) / sigma   # (n, mu)
            rank_mu_update = np.zeros((n, n))
            for i in range(mu):
                rank_mu_update += weights[i] * np.outer(dev[:, i], dev[:, i])
            C += cmu * rank_mu_update

            # Enforce symmetry (numerical stability)
            C = np.triu(C) + np.triu(C, 1).T

        # After budget exhausted, if there are remaining evaluations (fewer than lam), evaluate them randomly
        remaining = self.budget - evals
        if remaining > 0:
            extra_points = np.random.uniform(lb[:, np.newaxis], ub[:, np.newaxis], (n, remaining))
            for k in range(remaining):
                x = extra_points[:, k]
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x.copy()

        return best_x, best_y
