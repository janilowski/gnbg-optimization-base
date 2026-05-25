import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm is a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for black-box minimization. It adapts the search distribution (mean, covariance matrix, step size) based on successful steps.
# Search state: The algorithm maintains a multivariate normal distribution characterized by a centroid (mean), a covariance matrix (C), and a global step size (sigma). It also stores evolution paths for cumulative step-size adaptation and covariance matrix adaptation.
# Candidate generation: Each generation, lambda candidates are sampled from N(mean, sigma^2 * C). The candidates are then clipped to the search bounds to ensure feasibility.
# Selection and replacement: The top mu (mu = floor(lambda/2)) candidates are selected based on their objective function value (lower is better). The new mean is a weighted average of the selected points. The covariance matrix and step size are updated using the evolution paths and the selected points.
# Adaptation: The covariance matrix is updated using the rank-mu update and the rank-one update with evolution path. The step size is updated using cumulative step-size adaptation (CSA). The learning rates are controlled by standard CMA-ES parameters derived from the problem dimension.
# Exploration mechanisms: Sampling from a multivariate normal distribution with adaptively shaped covariance matrix allows anisotropic exploration. The step size adaptation ensures that the algorithm can increase or decrease the overall scale of the search.
# Exploitation mechanisms: As the selection pressure focuses on the best candidates, the covariance matrix is adapted to align with the local contour of the objective function, and the step size decreases as the optimum is approached, enabling fine-grained search.
# Boundary handling: Candidates that violate the box constraints are projected onto the boundary (clipped). This is simple but may cause clustering on boundaries; however, for many problems it works adequately.
# Budget strategy: The algorithm evaluates exactly lambda candidates per generation until the remaining budget is less than lambda, then evaluates a smaller final generation. The algorithm will continue to generate generations until budget is exhausted or it early stops due to too small sigma (in which case it uses remaining budget to sample uniformly).
# Closest known influences: This is a standard (mu, lambda)-CMA-ES, closely following Hansen's implementation, without restart strategies and with boundary clipping.
# Novelty or unusual aspects: None; it's a straightforward CMA-ES adaptation. The only deviation is the early stopping when sigma becomes extremely small, switching to random search for remaining budget.
# Failure modes: The algorithm may fail on highly multimodal landscapes because CMA-ES can converge to local optima. Boundary clipping may distort distribution near bounds. The step size may stagnate, but the fallback random search helps. The algorithm also assumes noiseless objective function.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # --- read bounds ----------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot determine bounds from func")

        lb = np.ravel(lb)
        ub = np.ravel(ub)
        dim = self.dim
        if len(lb) == 1 and dim > 1:
            lb = np.full(dim, lb[0])
            ub = np.full(dim, ub[0])
        elif len(lb) != dim:
            raise ValueError("Bounds dimension mismatch")

        budget = self.budget

        # --- CMA-ES parameters ----------------------------------------------
        lam = int(4 + 3 * np.log(dim))                 # population size
        mu = int(lam / 2)                              # parent number
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()              # recombination weights
        mueff = 1.0 / np.sum(weights ** 2)             # effective selection mass

        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)   # cumulation for C
        cs = (mueff + 2) / (mueff + dim + 5)                   # cumulation for sigma
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)                 # rank-one learning rate
        cmu = min(1 - c1,
                  2 * (mueff - 2 + 1.0 / mueff) / ((dim + 2) ** 2 + mueff))  # rank-mu learning rate
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        # --- initial state --------------------------------------------------
        mean = lb + np.random.rand(dim) * (ub - lb)    # centroid
        sigma = 0.5 * (ub - lb).mean()                 # global step size
        pc = np.zeros(dim)                             # evolution path for C
        ps = np.zeros(dim)                             # evolution path for sigma
        C = np.eye(dim)                                # covariance matrix
        B = np.eye(dim)                                # eigenvectors of C
        D = np.ones(dim)                               # sqrt(eigenvalues) of C

        # initial best point
        best_x = mean.copy()
        best_y = func(best_x)
        evals = 1

        # --- main loop ------------------------------------------------------
        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break

            pop_size = min(lam, remaining)             # last generation may be partial

            # sample candidates
            Z = np.random.randn(dim, pop_size)         # standard normal
            Y = B @ np.diag(D) @ Z
            samples = mean[:, np.newaxis] + sigma * Y
            samples = np.clip(samples, lb[:, np.newaxis], ub[:, np.newaxis])

            # evaluate
            y = np.empty(pop_size)
            for i in range(pop_size):
                x = samples[:, i]
                y[i] = func(x)
                evals += 1
                if y[i] < best_y:
                    best_y = y[i]
                    best_x = x.copy()
                if evals >= budget:
                    break

            # if budget exhausted or partial generation with < mu candidates, stop
            if evals >= budget or pop_size < mu:
                break

            # sort by fitness
            idx = np.argsort(y)
            best_idx = idx[:mu]
            sel = samples[:, best_idx]                 # selected parents

            # update mean
            mean_old = mean.copy()
            mean = sel @ weights

            # update evolution paths
            diff = (mean - mean_old) / sigma

            # ps (step-size path)
            inv_sqrt_C = B @ np.diag(1.0 / D) @ B.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrt_C @ diff)

            # pc (covariance path) – always update (hsig=1)
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * diff

            # update covariance matrix (rank-one + rank-mu)
            C = (1 - c1 - cmu) * C \
                + c1 * np.outer(pc, pc) \
                + cmu * (weights * (sel - mean_old[:, np.newaxis]) / sigma) @ \
                        ((sel - mean_old[:, np.newaxis]) / sigma).T

            # enforce symmetry
            C = np.triu(C) + np.triu(C, 1).T

            # update step size
            ps_norm = np.linalg.norm(ps)
            expected_ps_norm = np.sqrt(dim)
            sigma *= np.exp((cs / damps) * (ps_norm / expected_ps_norm - 1))
            sigma = max(sigma, 1e-20)

            # re‑compute eigendecomposition
            try:
                D2, B = np.linalg.eigh(C)
                D = np.sqrt(np.clip(D2, 1e-20, None))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)

            # fallback: if sigma collapsed, switch to uniform random search
            if sigma < 1e-20:
                remaining = budget - evals
                if remaining > 0:
                    rnd = np.random.rand(dim, remaining) * (ub - lb)[:, np.newaxis] + lb[:, np.newaxis]
                    for i in range(remaining):
                        x = rnd[:, i]
                        val = func(x)
                        evals += 1
                        if val < best_y:
                            best_y = val
                            best_x = x.copy()
                break

        return best_x, best_y
