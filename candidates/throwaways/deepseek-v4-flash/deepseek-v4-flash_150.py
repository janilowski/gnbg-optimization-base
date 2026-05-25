import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a variant of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
#           for continuous black-box minimization. Uses default parameters from Hansen (2006).
# Search state: Maintains mean vector (estimate of optimum), a scalar step-size sigma,
#               covariance matrix C, evolution paths pc and ps, and eigen-decomposition (B, D) of C.
# Candidate generation: Each generation samples λ = 4+3*log(dim) candidate solutions from
#                       N(m, sigma^2*C). Samples are clipped to the feasible box.
# Selection and replacement: The top μ = λ//2 solutions are selected via weighted recombination
#                            to form the new mean. The best-ever solution is tracked.
# Adaptation: Covariance matrix C updated with rank-one (pc) and rank-μ (selected steps) updates.
#             Step-size sigma adapted via cumulative step-size (CSA) using ps.
# Exploration mechanisms: Full multivariate normal sampling enables direction learning.
#                          CSA prevents premature convergence.
# Exploitation mechanisms: Weighted recombination and covariance learning accelerate progress
#                          toward promising regions.
# Boundary handling: Candidate points are clipped to lower/upper bounds.
# Budget strategy: Runs full generations while remaining budget permits; otherwise stops to
#                  never exceed the budget.
# Closest known influences: Standard CMA-ES (Hansen, Ostermeier, Müller); cmaes package.
# Novelty or unusual aspects: None – straightforward implementation with clipping.
# Failure modes: May stagnate if population is too small for high dimensions; clipping can
#                distort the distribution near boundaries; no explicit noise handling.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ---- extract bounds ----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        dim = self.dim
        budget = self.budget
        evals = 0

        # ---- initialise CMA-ES ----
        N = dim
        m = (lb + ub) / 2.0                     # initial mean
        range_ = ub - lb
        sigma = 0.2 * np.mean(range_)          # initial step-size

        lambda_ = int(4 + 3 * np.log(N))        # population size
        mu = lambda_ // 2                       # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()                # recombination weights
        mu_eff = 1.0 / np.sum(weights ** 2)

        # adaptation parameters (Hansen 2006 default)
        cc = (4.0 + mu_eff / N) / (N + 4.0 + 2.0 * mu_eff / N)
        cs = (mu_eff + 2.0) / (N + mu_eff + 5.0)
        c1 = 2.0 / ((N + 1.3) ** 2 + mu_eff)
        cmu = min(1.0 - c1,
                  2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((N + 2.0) ** 2 + mu_eff))
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (N + 1.0)) - 1.0) + cs

        pc = np.zeros(N)        # evolution path for C
        ps = np.zeros(N)        # evolution path for sigma
        C = np.eye(N)           # covariance matrix
        B = np.eye(N)           # eigenvectors of C (will be updated)
        D = np.ones(N)          # sqrt(eigenvalues) of C

        # ---- evaluate initial mean ----
        best_x = m.copy()
        best_y = func(best_x)
        evals = 1

        # ---- main loop ----
        while evals < budget:
            # 1. how many candidates can we generate?
            popsize = min(lambda_, budget - evals)
            if popsize < 1:
                break

            # 2. eigen-decomposition of C (for sampling and C^{-1/2})
            #    (we always recompute; okay for moderate dimensions)
            eigenvals, B = np.linalg.eigh(C)
            eigenvals = np.maximum(eigenvals, 1e-20)   # avoid zero
            D = np.sqrt(eigenvals)                     # sqrt of eigenvalues

            # 3. sample λ candidates
            z = np.random.randn(N, popsize)            # standard normal
            x = m[:, np.newaxis] + sigma * (B @ (D[:, np.newaxis] * z))

            # 4. clip to bounds
            for j in range(popsize):
                x[:, j] = np.clip(x[:, j], lb, ub)

            # 5. evaluate
            fits = np.array([func(x[:, i]) for i in range(popsize)])
            evals += popsize

            # 6. update best ever
            best_idx = np.argmin(fits)
            if fits[best_idx] < best_y:
                best_y = fits[best_idx]
                best_x = x[:, best_idx].copy()

            # if we didn't produce a full population, we cannot update mean / cov
            if popsize < lambda_:
                break

            # 7. sort and select top mu
            order = np.argsort(fits)
            x_sorted = x[:, order]
            m_old = m.copy()

            # 8. recombination -> new mean
            m = m_old + sigma * (x_sorted[:, :mu] - m_old[:, np.newaxis]) @ weights

            # 9. step difference (used for evolution paths)
            step = (m - m_old) / sigma

            # 10. compute C^{-1/2} (= B * diag(1/D) * B^T)
            D_inv = 1.0 / D
            C_inv_half = B @ np.diag(D_inv) @ B.T

            # 11. update ps
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mu_eff) * (C_inv_half @ step)

            # 12. hsig correction
            gen = evals // lambda_
            hsig = np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * gen)) < 1.4 + 2.0 / (N + 1.0)

            # 13. update pc
            if hsig:
                pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc) * mu_eff) * step
            else:
                pc = (1.0 - cc) * pc

            # 14. update covariance matrix (rank-one + rank-mu)
            artmp = (x_sorted[:, :mu] - m_old[:, np.newaxis]) / sigma
            C = (1.0 - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C) \
                + cmu * (artmp @ np.diag(weights) @ artmp.T)

            # 15. enforce symmetry
            C = (C + C.T) / 2.0

            # 16. adapt step-size sigma
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp((cs / damps) * (ps_norm / np.sqrt(N) - 1.0))

        return best_x, best_y
