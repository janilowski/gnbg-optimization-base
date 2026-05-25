import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact version of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for continuous black-box minimization. Uses default parameters based on Hansen's recommendations. Handles bound constraints by clamping.
# Search state: The search distribution is represented by a Gaussian with mean vector m and covariance matrix C, along with global step-size sigma. The algorithm also maintains evolution paths pc and ps for adaptation.
# Candidate generation: At each generation, lambda offspring are sampled from the Gaussian: x = m + sigma * y, where y ~ N(0, C). The eigendecomposition of C is computed to generate samples efficiently.
# Selection and replacement: The best mu offspring (mu = floor(lambda/2)) are selected and recombined using weighted averages to form the new mean. The covariance matrix and step-size are updated based on the selected points and evolution paths.
# Adaptation: Covariance matrix adaptation updates both the rank-1 (using evolution path pc) and rank-mu (using selected steps) components. Step-size adaptation uses cumulative step-size (CSA) with evolution path ps, which compares the length of ps to its expected value under a random selection.
# Exploration mechanisms: The Gaussian distribution with dynamic covariance allows anisotropic exploration. The step-size sigma controls overall scale.
# Exploitation mechanisms: Selection and recombination focus the search on promising regions. Covariance adaptation shapes the distribution to the local landscape.
# Boundary handling: Candidate solutions exceeding bounds are truncated (clamped) to the feasible region before evaluation. No repair or reflection is applied to preserve simplicity.
# Budget strategy: The population size lambda is set based on dimension. The algorithm runs full generations until remaining evaluations are less than lambda, then runs a final truncated generation to use all remaining budget. The distribution is not updated after the truncated generation.
# Closest known influences: Standard CMA-ES as described by Hansen (2006, 2016). Implementation follows the pycma package and Hansen's CMA-ES tutorial.
# Novelty or unusual aspects: Minimal modifications; clamping for bounds is used instead of more sophisticated handling. Adaptive population size is not implemented.
# Failure modes: May struggle with highly multimodal or discontinuous landscapes. Clamping can bias the distribution and slow convergence near boundaries. Small budget may not allow enough generations for adaptation.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            if hasattr(b, 'lb') and hasattr(b, 'ub'):
                lb = np.array(b.lb, dtype=float)
                ub = np.array(b.ub, dtype=float)
            else:
                raise ValueError("Bounds not found in func.bounds")
        else:
            raise ValueError("Cannot locate lower/upper bounds")

        dim = self.dim
        if lb.shape[0] != dim:
            lb = lb[:dim]
            ub = ub[:dim]

        # CMA-ES parameter settings
        lambda_ = int(4 + 3 * np.log(dim))          # population size
        lambda_ = max(lambda_, 2)                   # ensure at least 2
        lambda_ = min(lambda_, self.budget)         # respect budget
        mu = max(1, lambda_ // 2)                   # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)            # variance effective selection mass

        # Adaptation parameters (Hansen's defaults)
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2.0 / ((dim + 1.3)**2 + mueff)
        cmu = min(1.0 - c1,
                  2.0 * (mueff - 2.0 + 1.0/mueff) / ((dim + 2.0)**2 + mueff))
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff-1)/(dim+1)) - 1.0) + cs

        # Initialisation
        m = 0.5 * (lb + ub)                         # mean
        sigma = 0.2 * np.mean(ub - lb)               # isotropic step size
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim)                              # eigenvectors
        D = np.ones(dim)                             # sqrt(eigenvalues)
        evals = 0
        best_y = np.inf
        best_x = None

        def sample():
            """Draw a single candidate from the current distribution."""
            z = np.random.randn(dim)
            y = B @ (D * z)                          # y ~ N(0, C)
            x = m + sigma * y
            return np.clip(x, lb, ub)                # clamp to bounds

        while evals < self.budget:
            remaining = self.budget - evals
            # Use a full generation if possible, otherwise a truncated one
            if remaining >= lambda_:
                current_lambda = lambda_
                do_update = True
            else:
                current_lambda = remaining
                do_update = False

            # Sample and evaluate
            X = np.array([sample() for _ in range(current_lambda)])
            Y = np.array([func(x) for x in X])
            evals += current_lambda

            # Update best observed solution
            idx_min = np.argmin(Y)
            if Y[idx_min] < best_y:
                best_y = Y[idx_min]
                best_x = X[idx_min].copy()

            # If this was not a full generation, stop (budget exhausted)
            if not do_update or evals >= self.budget:
                break

            # Selection and recombination
            sorted_idx = np.argsort(Y)
            X_selected = X[sorted_idx[:mu]]
            m_old = m.copy()
            m = np.dot(weights, X_selected)

            # Selected steps in the original coordinate system
            y_selected = (X_selected - m_old) / sigma

            # Update evolution paths
            invsqrtC = B @ np.diag(1.0 / D) @ B.T   # C^(-1/2)
            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ (m - m_old) / sigma)
            pc = (1-cc)*pc + np.sqrt(cc*(2-cc)*mueff) * (m - m_old) / sigma

            # Update covariance matrix
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
            for i in range(mu):
                C += cmu * weights[i] * np.outer(y_selected[i], y_selected[i])
            C = np.triu(C) + np.triu(C, 1).T        # enforce symmetry

            # Update step size
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp((cs/damps) * (norm_ps / np.sqrt(dim) - 1))

            # Re‑compute eigendecomposition
            try:
                eigvals, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(eigvals, 1e-20))
            except np.linalg.LinAlgError:
                pass   # keep previous B, D

        return best_x, best_y
