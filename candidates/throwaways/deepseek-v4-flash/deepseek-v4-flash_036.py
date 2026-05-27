import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for black-box minimization. It adapts the search distribution's mean, covariance, and step-size using the classic (μ,λ) selection with weighted recombination. The algorithm aims to efficiently optimize low-to-moderate dimensional functions within a limited evaluation budget.
# Search state: Maintains a multivariate normal distribution defined by mean vector (mean), covariance matrix (C), and global step size (sigma). Also maintains evolution paths for covariance (pc) and step-size (ps) to exploit correlations and control adaptation.
# Candidate generation: In each generation, λ candidate solutions are sampled from N(mean, sigma^2 * C). The eigenvectors and eigenvalues of C are computed via eigh. Candidates are generated as mean + sigma * (eigenvectors @ (randn * sqrt(eigenvalues))). After generation, each candidate is clamped to the variable bounds.
# Selection and replacement: Truncation selection: the μ best candidates (μ = λ/2) are selected based on fitness. Offspring replace the entire population each generation (no elitism via separate archive, but the best overall solution is tracked and returned). Weighted recombination computes the new mean as a weighted average of the μ best solutions.
# Adaptation: Two evolution paths accumulate information over generations. The step-size sigma is updated based on the length of the conjugate evolution path ps. The covariance matrix C is updated using rank-one update (cumulative step-size adaptation) and rank-μ update (weighted covariance of selected steps). Learning rates c1, cmu, cc, csigma are set using default formulas from Hansen.
# Exploration mechanisms: The isotropic initial distribution and stochastic sampling provide exploration. The covariance adaptation can discover and exploit correlated variable spaces. The step-size adaptation prevents premature convergence.
# Exploitation mechanisms: The mean moves toward promising areas via weighted recombination. The covariance matrix learns the principal directions of the search space, focusing sampling in the most promising region.
# Boundary handling: Candidate solutions that exceed the lower or upper bounds are clamped (simple clip). This is a non-ideal but simple approach; it may cause bias near boundaries but works for many problems.
# Budget strategy: The algorithm stops when the remaining budget is less than the population size λ (i.e., cannot complete another full generation). The best candidate found so far is returned. The population size is adjusted based on dimension but capped to avoid excessive evaluations per generation when budget is small (λ = min( default, budget//4)).
# Closest known influences: This code is a straightforward implementation of the classic CMA-ES algorithm (Hansen & Ostermeier, 2001) as described in Hansen's CMA-ES tutorial, with minor simplifications (e.g., fixed learning rates, no restarts, clamp boundary handling).
# Novelty or unusual aspects: None; the code is intended as a clean, compact reference implementation.
# Failure modes: May perform poorly on highly multimodal functions due to premature convergence when budget is small. The clamp boundary handling can cause the algorithm to become stuck near boundaries if the optimum is at the boundary and the sampling repeatedly gets clipped. For large dimensions (>>50), the eigendecomposition becomes expensive and the default population size grows, requiring a large budget. The algorithm does not handle noise or discrete variables.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Get variable bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            raise ValueError("No bounds found in function object")
        diff = ub - lb
        if np.any(diff <= 0):
            raise ValueError("Invalid bounds (lower >= upper in some dimensions)")

        n = self.dim
        budget = self.budget

        # Population size (default from CMA-ES, scaled to budget)
        lam = 4 + int(3 * np.log(n))
        lam = max(4, min(lam, budget // 4))   # ensure at least 4 and leaves room for generations
        if lam % 2 == 1:
            lam += 1
        mu = lam // 2                           # number of parents

        # Recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)

        # Learning rates (Hansen's defaults)
        csigma = (mueff + 2) / (n + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + csigma
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

        # Initialisation
        mean = lb + np.random.rand(n) * (ub - lb)   # random starting point within bounds
        sigma = 0.3 * np.mean(diff)                 # global step size
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        evals = 0

        # Evaluate initial point
        best_x = mean.copy()
        best_y = func(mean)
        evals += 1

        # Main loop: stop when the next generation would exceed the budget
        while evals + lam <= budget:
            # Eigendecomposition of C (stable, symmetric)
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            sqrt_eig = np.sqrt(eigenvalues)

            # Sample λ offspring
            pop = np.empty((lam, n))
            for i in range(lam):
                z = np.random.randn(n)
                y = eigenvectors @ (z * sqrt_eig)          # N(0, C)
                x = mean + sigma * y
                x = np.clip(x, lb, ub)                    # boundary handling
                pop[i] = x

            # Evaluate offspring
            fits = np.array([func(x) for x in pop])
            evals += lam

            # Sort by fitness
            order = np.argsort(fits)
            pop_sorted = pop[order]
            fits_sorted = fits[order]

            # Update overall best
            if fits_sorted[0] < best_y:
                best_y = fits_sorted[0]
                best_x = pop_sorted[0].copy()

            # Update mean
            old_mean = mean.copy()
            mean = weights @ pop_sorted[:mu, :]

            # Normalised step (z = C^(-1/2) * (mean - old_mean) / sigma)
            delta = (mean - old_mean) / sigma
            zmean = eigenvectors @ ((eigenvectors.T @ delta) / sqrt_eig)

            # Evolution paths
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * zmean
            ps = (1 - csigma) * ps + np.sqrt(csigma * (2 - csigma) * mueff) * zmean

            # Step size adaptation
            ps_norm = np.linalg.norm(ps)
            sigma *= np.exp((csigma / damps) * (ps_norm / np.sqrt(n) - 1))

            # Covariance matrix adaptation (rank-one + rank-mu)
            y = (pop_sorted[:mu, :] - old_mean) / sigma
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])
            C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            C = (C + C.T) / 2.0   # enforce symmetry

        return best_x, best_y
