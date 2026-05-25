import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
#          for black-box minimization. Uses a Gaussian search distribution with
#          iterative adaptation of mean, covariance, and step size.
# Search state: mean vector m, covariance matrix C, step size sigma, and two
#               evolution paths pc and ps (cumulative step-size adaptation).
# Candidate generation: Each generation, lambda points are sampled from a
#   multivariate normal N(m, sigma**2 * C) using a Cholesky decomposition of C.
#   Offspring are reflected at the boundary to stay within bounds.
# Selection and replacement: All lambda individuals are evaluated; the mu best
#   (mu = floor(lambda/2) by default but weighted) are recombined to form the
#   new mean. The covariance update uses both rank-1 (evolution path) and
#   rank-mu (selected differences) terms.
# Exploration mechanisms: Step-size control via cumulative path length (ps)
#   keeps sigma from collapsing prematurely; random sampling ensures diversity.
# Exploitation mechanisms: Covariance learning adapts the shape of the search
#   distribution to the local landscape, focusing sampling along promising
#   directions.
# Boundary handling: Reflective mirroring: if a coordinate falls outside the
#   box, it is reflected back inside by the excess distance. This avoids
#   concentration at the boundary and keeps evaluations inside the domain.
# Budget strategy: The initial mean is evaluated once, then we run as many
#   full generations (each consuming lambda evaluations) as the budget allows.
#   Any remaining evaluations (less than lambda) are used for additional random
#   sampling without further adaptation.
# Closest known influences: Classic CMA-ES (Hansen & Ostermeier, 2001) with
#   default population size lambda = 4 + floor(3*log(n)).
# Novelty or unusual aspects: None; this is a straightforward implementation
#   with reflective boundary handling, bundled into a single class with no
#   external dependencies beyond numpy.
# Failure modes: May struggle on highly multimodal or extremely high‑dimensional
#   landscapes if the budget is low relative to dimension. Reflective handling
#   can distort the distribution near boundaries if bounds are very tight.
#   Fixed population size may be suboptimal for extremely small budgets.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """CMA-ES minimizer for GNBG black-box functions."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # --- bounds ---
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lo = np.asarray(func.lower, dtype=float)
            hi = np.asarray(func.upper, dtype=float)
        else:
            lo = np.asarray(func.bounds.lb, dtype=float)
            hi = np.asarray(func.bounds.ub, dtype=float)
        n = self.dim
        bud = self.budget

        # --- default CMA-ES parameters ---
        lam = int(4 + 3 * np.log(n))          # population size
        mu = lam // 2                         # number of parents
        # recombination weights (standard log weights)
        weights = np.log((lam + 1) / 2) - np.log(np.arange(1, lam + 1))
        weights[:mu] /= np.sum(weights[:mu])   # only mu best used later
        mu_eff = 1.0 / np.sum(weights[:mu]**2)

        # learning rates (standard settings)
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        c1 = 2.0 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2)**2 + mu_eff))
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs

        # --- initialisation ---
        rng = np.random.default_rng()          # harness sets seed
        m = rng.uniform(lo, hi, size=n)        # mean vector
        sigma = 0.3 * (hi - lo).mean() / 2     # initial step size
        C = np.eye(n)                          # covariance matrix
        pc = np.zeros(n)
        ps = np.zeros(n)

        # --- evaluate starting point ---
        best_x = m.copy()
        best_y = func(best_x)
        evals = 1

        # --- helper: reflect a point into bounds ---
        def reflect(x):
            # reflect each coordinate that is outside
            too_low = x < lo
            too_high = x > hi
            x[too_low] = 2 * lo[too_low] - x[too_low]
            x[too_high] = 2 * hi[too_high] - x[too_high]
            # a second reflection may be needed if the point overshoots
            x = np.clip(x, lo, hi)   # safety clip after reflection
            return x

        # --- main loop ---
        while evals < bud:
            # decide how many points to generate this generation
            lam_cur = min(lam, bud - evals)
            if lam_cur < 2:               # not enough for meaningful sampling
                # try one extra random point
                x_try = rng.uniform(lo, hi, size=n)
                y_try = func(x_try)
                evals += 1
                if y_try < best_y:
                    best_x, best_y = x_try.copy(), y_try
                break

            # generate offspring
            try:
                B = np.linalg.cholesky(C)   # B * B^T = C
            except np.linalg.LinAlgError:
                # if C becomes ill‑conditioned, reset to identity
                C = np.eye(n)
                B = np.eye(n)

            pop = np.empty((lam_cur, n))
            fit = np.empty(lam_cur)
            for i in range(lam_cur):
                z = rng.normal(0, 1, size=n)
                x = m + sigma * B @ z
                x = reflect(x)
                pop[i] = x
                fit[i] = func(x)
                evals += 1
                if fit[i] < best_y:
                    best_x, best_y = x.copy(), fit[i]

            # update CMA state only if we have a full generation
            if lam_cur == lam:
                # sort by fitness (minimization)
                idx = np.argsort(fit)
                pop_sorted = pop[idx]
                # update mean
                m_old = m.copy()
                m = np.dot(weights[:mu], pop_sorted[:mu])

                # update evolution paths
                invsqrtC = np.linalg.inv(B)
                z_mean = (m - m_old) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * invsqrtC @ z_mean
                ps_norm = np.linalg.norm(ps)
                hs = 1 if ps_norm / np.sqrt(1 - (1 - cs)**(2 * evals / lam)) < (1.4 + 2 / (n + 1)) else 0
                delta_hs = (1 - hs) * c1 * cc * (2 - cc)
                pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * z_mean

                # update covariance matrix
                C = (1 - c1 - cmu) * C \
                    + c1 * (pc[:, None] * pc[None, :] + delta_hs * C) \
                    + cmu * np.dot((pop_sorted[:mu] - m_old).T,
                                   np.diag(weights[:mu]) @ (pop_sorted[:mu] - m_old)) / sigma**2

                # update step size
                sigma *= np.exp((cs / damps) * (ps_norm / (np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))) - 1))

                # enforce positive definiteness (simple safeguard)
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-12:
                    C += 1e-12 * np.eye(n)

            # if not full generation, we still evaluated points; stop after
            # this partial generation (further adaptation is unreliable)
            if lam_cur < lam:
                break

        return best_x, best_y
