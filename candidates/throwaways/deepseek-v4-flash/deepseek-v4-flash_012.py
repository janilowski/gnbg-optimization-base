import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a (μ,λ)-CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
# for black-box minimization. It is designed to be compact yet robust across a wide range of
# dimensions and function landscapes.
#
# Search state: The algorithm maintains a multivariate normal distribution defined by its
# mean vector (center of mass), an overall step size σ, and a symmetric positive definite
# covariance matrix C. Two evolution paths pσ and pc store low‑pass filtered information
# about the steps taken to adapt σ and C respectively.
#
# Candidate generation: Each generation, λ = popsize candidate solutions are sampled from
# the current distribution: x = mean + σ * Cholesky(C) * z, where z ~ N(0,I). New candidates
# are evaluated immediately.
#
# Selection and replacement: After evaluation, the μ best candidates (μ = λ//2) are selected.
# The mean is updated as the uniformly weighted centroid of those μ candidates (truncation
# selection). The covariance matrix is updated using both the rank‑1 update from the
# evolution path pc and the rank‑μ update from the selected steps.
#
# Adaptation: Step size σ is adapted via cumulative step‑size adaptation (CSA), using the
# conjugate evolution path pσ. The covariance matrix is updated with standard CMA‑ES
# learning rates c1 and cmu. The damping parameter damps controls the speed of σ adaptation.
#
# Exploration mechanisms: Large σ and high‑variance eigenvectors of C encourage broad
# sampling. The rank‑μ update reinforces successful search directions, while the CSA
# mechanism prevents premature convergence of σ.
#
# Exploitation mechanisms: As the covariance matrix adapts to the local landscape, the
# sampling distribution becomes increasingly aligned with promising directions. The
# centroid of the best μ points acts as a low‑pass filter that drives the mean toward
# optima.
#
# Boundary handling: Out‑of‑bounds coordinates are reflected back into the feasible domain
# using mirroring. If mirroring would still leave a coordinate outside (e.g. when the
# violated region is larger than the box), the coordinate is clipped to the nearest bound.
#
# Budget strategy: The number of function evaluations is tracked globally. The algorithm
# stops as soon as the total evaluations exceed the budget. The best objective value
# found so far is always remembered.
#
# Closest known influences: Standard CMA‑ES as described by Hansen (2006) and implemented
# in the cma package. This code uses uniform recombination weights and a fixed population
# size schedule based on the problem dimension.
#
# Novelty or unusual aspects: The implementation is intentionally minimalist – it uses
# uniform weights, omits the weighted recombination from full CMA‑ES, and performs the
# eigendecomposition every generation (instead of using a Cholesky update) for simplicity.
#
# Failure modes: On extremely ill‑conditioned or highly multimodal landscapes, the fixed
# population size may be insufficient. The algorithm may stall when σ becomes very small
# in a narrow valley. Very high dimensions (e.g., >100) may require tuning the population
# size and learning rates.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize CMA-ES optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of function evaluations.
        dim : int
            Problem dimensionality.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the CMA-ES optimizer on the given function.

        Parameters
        ----------
        func : obj
            Objective function. Must have either `lower`/`upper` attributes
            or `bounds.lb`/`bounds.ub` arrays.

        Returns
        -------
        best_x : np.ndarray
            Best found solution.
        best_y : float
            Objective value at best_x.
        """
        # ---- read bounds -------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lo = np.asarray(func.lower, dtype=float)
            hi = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lo = np.asarray(func.bounds.lb, dtype=float)
            hi = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds on func")

        dim = self.dim
        lo = lo.flatten()
        hi = hi.flatten()
        # ensure bounds are valid
        if np.any(lo >= hi):
            raise ValueError("Lower bounds must be strictly less than upper bounds")

        # ---- CMA-ES parameters ------------------------------------------
        n = dim
        # population size: typical schedule
        popsize = max(4, int(4 + 3 * np.log(n)))
        mu = popsize // 2
        lam = popsize

        # strategy parameters (standard values)
        mu_eff = mu  # uniform weights => sum(wi)^2/sum(wi^2) = mu
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs

        # initial mean: center of bounds
        mean = (lo + hi) / 2.0
        # initial step size: 0.3 * average range
        sigma = 0.3 * np.mean(hi - lo)
        # covariance matrix and eigendecomposition
        C = np.eye(n)
        # eigen decomposition: C = B * D^2 * B^T
        B = np.eye(n)
        D = np.ones(n)

        # evolution paths
        pc = np.zeros(n)
        ps = np.zeros(n)

        # state for best so far
        best_x = mean.copy()
        best_y = np.inf
        evals = 0

        # helper: boundary handling (mirror)
        def mirror(x_in):
            x = x_in.copy()
            for i in range(n):
                if x[i] < lo[i]:
                    d = lo[i] - x[i]
                    x[i] = lo[i] + d
                    if x[i] > hi[i]:
                        x[i] = hi[i]
                elif x[i] > hi[i]:
                    d = x[i] - hi[i]
                    x[i] = hi[i] - d
                    if x[i] < lo[i]:
                        x[i] = lo[i]
            return x

        # ---- main loop --------------------------------------------------
        while evals < self.budget:
            # sample new population
            pop = np.zeros((lam, n))
            fit = np.full(lam, np.inf)
            for i in range(lam):
                z = np.random.randn(n)
                x = mean + sigma * (B @ (D * z))   # Cholesky-like transformation
                x = mirror(x)
                pop[i] = x
                # evaluate
                y = func(x)
                evals += 1
                fit[i] = y
                # keep best overall
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                # if budget exhausted during generation, break early
                if evals >= self.budget:
                    return best_x, best_y

            # selection: sort by fitness, pick best mu
            idx = np.argsort(fit)
            pop_best = pop[idx[:mu]]
            fit_best = fit[idx[:mu]]  # not used further

            # update mean (uniform weights)
            mean_old = mean.copy()
            mean = np.mean(pop_best, axis=0)

            # compute step differences (selected individuals)
            # For update of C we need the vectors (x - mean_old) / sigma
            diff = (pop_best - mean_old) / sigma
            diff_mean = (mean - mean_old) / sigma

            # update evolution paths
            # ps: conjugate evolution path
            # transform diff_mean to principal components space
            invsqrtC = B @ np.diag(1.0 / D) @ B.T   # C^(-1/2)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ diff_mean)

            # pc: anisotropic evolution path
            hsig = (np.linalg.norm(ps) /
                    np.sqrt(1 - (1 - cs)**(2 * evals / lam)) /
                    1.4 + 1.0) < 2.0   # heuristics
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * diff_mean

            # update covariance matrix
            # rank-1 update
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) +
                                             (1 - hsig) * cc * (2 - cc) * C)
            # rank-mu update
            for j in range(mu):
                C += cmu * np.outer(diff[j], diff[j]) / mu

            # enforce symmetry
            C = np.triu(C) + np.triu(C, 1).T
            # ensure positive definiteness: add small ridge
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-20:
                C += 1e-20 * np.eye(n)

            # update step size sigma
            chi = np.sqrt(n) * (1 - 1.0/(4*n) + 1.0/(21*n*n))
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi - 1))

            # eigendecomposition of C
            try:
                D2, B = np.linalg.eigh(C)
                # D = sqrt(eigenvalues)
                D = np.sqrt(np.maximum(D2, 0.0))
                # ensure positive D (clamp small negative due to numerical errors)
            except np.linalg.LinAlgError:
                # fallback if decomposition fails: keep previous B, D
                pass

            # optional: rescale sigma if D very small? Not done here.

        # budget exhausted
        return best_x, best_y
