import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simplified CMA-ES (Covariance Matrix Adaptation Evolution Strategy) that adapts
#          a multivariate normal distribution to explore the search space, combined with
#          adaptive step-size control and restarts upon convergence.
# Search state: Mean vector, covariance matrix (as Cholesky factor L), step-size sigma,
#               evolution paths (pc, ps), and best-found solution (xopt, fopt).
# Candidate generation: Samples from N(mean, sigma^2 * L @ L.T) via L @ z + mean,
#                      where z ~ N(0, I). Population size lambda adapts with dimension.
# Selection and replacement: Keep top mu samples (elitism), weighted recombination of
#                           selected points to update mean, update covariance from
#                           rank-mu updates and evolution path.
# Adaptation: CMA-ES-style path updates for step-size (ps) and covariance (pc);
#             step-size adapts via success-based exponential rule.
# Exploration mechanisms: Large initial sigma explores broadly; covariance adaptation
#                         handles non-separable, scaled, and rotated landscapes.
# Exploitation mechanisms: Weighted recombination focuses on best solutions; shrinking
#                          sigma during convergence intensifies local search.
# Boundary handling: Clamp all candidates to [lower, upper] bounds.
# Budget strategy: Spend lambda evaluations per iteration; restart if converged or
#                 if no improvement for 10*dim evaluations; cap restarts to avoid
#                 infinite loops within budget.
# Closest known influences: Hansen & Ostermeier (2001) CMA-ES; simplified without
#                          active covariance or threshold sequential restarts.
# Novelty or unusual aspects: Uses Cholesky decomposition for stable sampling,
#                             handles singular covariance via re-initialization,
#                             median-based step-size adaptation for robustness.
# Failure modes: May underperform on highly irregular functions; relies on
#                bounded search space; Cholesky failure rare but handled.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initialize optimizer.

        Args:
            budget: Maximum number of function evaluations.
            dim: Dimensionality of the search space.
        """
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        """
        Optimize the given objective function.

        Args:
            func: Black-box objective with .lower/.upper or .bounds.lb/.bounds.ub.

        Returns:
            Tuple of (best_x, best_y) where best_x is the best solution found
            and best_y is its objective value.
        """
        # --- Extract bounds ---
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: [-5, 5]^dim (common benchmark range)
            lower = np.full(self.dim, -5.0)
            upper = np.full(self.dim, 5.0)

        # Ensure arrays are 1-D with correct length
        lower = np.atleast_1d(lower).astype(float)
        upper = np.atleast_1d(upper).astype(float)
        if lower.shape[0] != self.dim:
            lower = np.full(self.dim, lower[0] if lower.size == 1 else lower.mean())
        if upper.shape[0] != self.dim:
            upper = np.full(self.dim, upper[0] if upper.size == 1 else upper.mean())

        # --- CMA-ES parameters (self-adaptive) ---
        lam = max(16, int(4 + 3 * np.log(self.dim)))   # population size
        mu = lam // 2                                   # parents for recombination
        # Recombination weights (positive, sum to 1)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()              # normalized
        mu_eff = 1.0 / (weights @ weights)             # variance effective selection mass

        # Adaptation constants (close to standard CMA-ES defaults)
        cc = (4 + self.dim) / (self.dim * (4 + 2 * self.dim / lam) + 4)
        cs = (self.dim + 2) / (self.dim + 5)
        cap = 1.0                                       # step-size damping
        c1 = 2.0 / ((self.dim + 1.3) ** 2 + mu_eff)   # rank-1 update weight
        cmu = min(1 - c1, 2 * (mu_eff - 1 + 2 / self.dim) / ((self.dim + 2) ** 2 + mu_eff))
        # Ensure cmu does not cause negative update (safety)
        cmu = max(cmu, 1e-10)

        # --- Initialize state ---
        mean = (lower + upper) / 2.0                    # initial mean
        sigma = (upper - lower).max() * 0.25            # initial step size
        C = np.eye(self.dim)                            # covariance (identity * sigma^2)
        pc = np.zeros(self.dim)                         # evolution path for C
        ps = np.zeros(self.dim)                         # evolution path for sigma
        L = np.linalg.cholesky(C)                       # Cholesky factor (C = L @ L.T)

        best_x = None
        best_y = float('inf')
        evals = 0
        restarts = 0
        max_restarts = 5 + self.dim // 10
        no_improvement = 0
        med_fitness_prev = float('inf')

        # --- Main loop ---
        while evals < self.budget:
            # --- Sample population ---
            try:
                # z ~ N(0, I); x = mean + sigma * L @ z
                z = np.random.randn(lam, self.dim)
                # Use L directly: x = mean + sigma * (z @ L.T) == (mean) + sigma * L @ z.T
                # Actually x_i = mean + sigma * (L @ z_i.T).T = mean + sigma * (z_i @ L.T). No:
                # Cholesky L satisfies C = L @ L.T. Sampling: L @ randn() + mean.
                pop = mean + sigma * (z @ L.T)
            except np.linalg.LinAlgError:
                # Rare: C becomes non-positive-definite; re-init C
                C = np.eye(self.dim) + 1e-6 * np.random.rand(self.dim, self.dim)
                C = (C + C.T) / 2
                L = np.linalg.cholesky(C)
                pop = mean + sigma * (np.random.randn(lam, self.dim) @ L.T)

            # Clip to bounds
            pop = np.clip(pop, lower, upper)

            # --- Evaluate ---
            fitness = np.empty(lam)
            for i in range(lam):
                if evals >= self.budget:
                    break
                fitness[i] = func(pop[i])
                evals += 1

            if evals >= self.budget:
                break

            # --- Update best solution ---
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_y:
                best_y = fitness[best_idx]
                best_x = pop[best_idx].copy()
                no_improvement = 0
            else:
                no_improvement += lam

            # --- Selection and recombination ---
            sort_idx = np.argsort(fitness)
            selected = pop[sort_idx[:mu]]
            # Weighted mean update
            mean_old = mean.copy()
            mean = np.sum(weights[:, np.newaxis] * selected, axis=0)

            # --- Update evolution paths ---
            # Cumulation for step-size (ps)
            # sqrt(mu_eff / (dim + 2)) is the expected norm of N(0, I) projected
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (mean - mean_old) / sigma

            # Indicator for hypothesis (use hsig to damp pc update)
            hsig = 1.0 if np.linalg.norm(ps) < 1.5 * np.sqrt(self.dim) else 0.0

            # Cumulation for covariance (pc)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - mean_old) / sigma

            # --- Update covariance matrix C ---
            # Rank-1 update from evolution path
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)

            # Rank-mu update from selected individuals
            for i in range(mu):
                diff = (selected[i] - mean_old) / sigma
                C += cmu * weights[i] * np.outer(diff, diff)

            # --- Enforce symmetry (numerical safety) ---
            C = (C + C.T) / 2

            # --- Update step-size sigma ---
            # Success rule: compare median fitness to previous median
            median_f = np.median(fitness)
            if median_f < med_fitness_prev:
                sigma *= np.exp(cs / cap * 0.2)  # increase step on success
            else:
                sigma *= np.exp(-cs / cap)       # decrease step on failure
            med_fitness_prev = median_f

            # Prevent sigma from collapsing
            sigma = max(sigma, 1e-10 * (upper - lower).max())

            # --- Update Cholesky factor L via SVD for stability ---
            # For simplicity, use eigendecomposition (available in numpy) and re-compute L
            # Eigvals are guaranteed non-negative due to symmetric update
            try:
                # Ensure positive definiteness by flooring eigenvalues
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                L = eigvecs * np.sqrt(eigvals) @ eigvecs.T  # L = V @ sqrt(D)
                # This yields L where L @ L.T = V @ D @ V.T = C (since C = V @ D @ V.T)
                # Actually: eigvecs columns are orthonormal, so eigvecs @ diag(eigvals) @ eigvecs.T = C
                # And L = eigvecs @ sqrt(diag(eigvals)) satisfies L @ L.T = C
                L = eigvecs * np.sqrt(eigvals)  # each column: eigenvector * sqrt(eigval)
                # Ensure L is lower-triangular? Eigvecs are not lower-triangular.
                # Better: L = eigvecs @ np.diag(np.sqrt(eigvals)). But eigvecs are orthogonal.
                # For sampling: L @ z = eigvecs @ (sqrt(eigvals) * (eigvecs.T @ z))
                # But we need L s.t. C = L @ L.T and L lower-triangular (Cholesky).
                # Use np.linalg.cholesky on (C + small_noise) to get proper L.
                L = np.linalg.cholesky(C + 1e-8 * np.eye(self.dim))
            except np.linalg.LinAlgError:
                # Fallback: reset to identity covariance
                C = np.eye(self.dim) * sigma**2
                L = np.eye(self.dim) * sigma

            # --- Restart logic ---
            # Restart if sigma is too small relative to bounds, or if stagnated
            range_scale = (upper - lower).max()
            tol = 1e-8 * range_scale
            if sigma < tol * 0.01 or no_improvement > 10 * self.dim:
                mean = lower + np.random.rand(self.dim) * (upper - lower)
                sigma = range_scale * 0.25
                C = np.eye(self.dim) * sigma**2
                pc = np.zeros(self.dim)
                ps = np.zeros(self.dim)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(self.dim)
                restarts += 1
                no_improvement = 0
                if restarts > max_restarts:
                    break

        return best_x if best_x is not None else mean, best_y
