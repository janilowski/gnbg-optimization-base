import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact implementation of the (mu,lambda)-CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
#          for continuous black-box minimization. It adapts the covariance matrix and global step size.
# Search state: A multivariate normal distribution defined by mean vector, step size sigma, and
#               covariance matrix C. Also maintains evolution paths p_sigma and p_c.
# Candidate generation: lambda candidate solutions are sampled as mean + sigma * sqrt(C) * z
#                       where z ~ N(0,I).
# Selection and replacement: The mu best candidates (by fitness) are used to update the mean and the
#                            covariance matrix via weighted recombination.
# Adaptation: The covariance matrix is updated using rank-one (p_c * p_c^T) and rank-mu updates.
#             The global step size sigma is updated using cumulative step-size adaptation (CSA)
#             based on the length of the evolution path p_sigma.
# Exploration mechanisms: The multivariate normal distribution with full covariance allows
#                         directed exploration along promising ridges. The step-size adaptation
#                         prevents premature convergence.
# Exploitation mechanisms: The mean is a weighted average of the best mu points, concentrating
#                         the search around the best region found so far.
# Boundary handling: Sampled points are clipped to the feasible box [lb, ub].
# Budget strategy: The algorithm stops as soon as the next generation would exceed the remaining
#                  budget. The population size lambda is capped to the budget if budget is very small.
# Closest known influences: Standard CMA-ES as described by Hansen (2006) with rank-one update.
# Novelty or unusual aspects: None; a straightforward textbook implementation.
# Failure modes: Poor performance on strongly multimodal landscapes if the population size is too small.
#                Clipping boundaries can distort the covariance structure.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Read bounds
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb, dtype=float)
                ub = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Cannot read bounds from function object")

        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)

        dim = self.dim
        budget = self.budget
        evaluations = 0

        # Initialization
        mean = lb + (ub - lb) * np.random.uniform(0, 1, size=dim)
        sigma = 0.2 * (ub - lb).mean()  # initial step size

        # Population size (grows slowly with dimension)
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        # Cap lambda to budget so we can have at least one generation
        lambda_ = min(lambda_, budget)
        mu = max(1, lambda_ // 2)

        # Recombination weights
        weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        weights /= weights.sum()
        mueff = 1.0 / (weights ** 2).sum()

        # Strategy parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4.0 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((dim + 2) ** 2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma

        # Evolution paths and covariance
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        # We will compute Cholesky factor each generation

        # Evaluate initial mean (one evaluation)
        best_x = mean.copy()
        best_y = func(mean)
        evaluations += 1

        # Main generation loop
        while evaluations < budget:
            # Determine if we have enough budget for a full generation
            if budget - evaluations < lambda_:
                # Not enough budget for full generation, do a final sample of remaining
                remaining = budget - evaluations
                if remaining == 0:
                    break
                # Sample remaining points
                z = np.random.randn(dim, remaining)
                # Compute Cholesky factor of C
                A = np.linalg.cholesky(C)
                sampling_points = mean[:, None] + sigma * (A @ z)
                sampling_points = np.clip(sampling_points, lb[:, None], ub[:, None])  # dim x remaining
                for i in range(remaining):
                    x = sampling_points[:, i]
                    y = func(x)
                    evaluations += 1
                    if y < best_y:
                        best_y = y
                        best_x = x.copy()
                break  # budget exhausted

            # Full generation sampling
            z_all = np.random.randn(dim, lambda_)  # columns = z vectors
            A = np.linalg.cholesky(C)
            sampling_points = mean[:, None] + sigma * (A @ z_all)  # dim x lambda
            sampling_points = np.clip(sampling_points, lb[:, None], ub[:, None])

            # Evaluate all points
            y_vals = np.empty(lambda_)
            for i in range(lambda_):
                x = sampling_points[:, i]
                y_vals[i] = func(x)
                evaluations += 1
                if y_vals[i] < best_y:
                    best_y = y_vals[i]
                    best_x = x.copy()

            # Selection: get indices of best mu points
            idx_sorted = np.argsort(y_vals)
            idx_sel = idx_sorted[:mu]

            # Update mean
            old_mean = mean.copy()
            mean = np.dot(weights, sampling_points[:, idx_sel].T)  # shape (dim,)

            # Update evolution paths
            # Weighted average of the selected z vectors (the direction in the normalised space)
            z_sel = z_all[:, idx_sel]  # dim x mu
            z_mean = np.dot(weights, z_sel.T)  # shape (dim,)

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_mean

            mean_step = (mean - old_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * mean_step

            # Covariance matrix update (rank-one + rank-mu)
            # Rank-mu sum: sum w_i * ( (x_i - old_mean)/sigma ) * ((x_i - old_mean)/sigma)^T
            x_diff = (sampling_points[:, idx_sel] - old_mean[:, None]) / sigma  # dim x mu
            rank_mu_update = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu_update += weights[i] * np.outer(x_diff[:, i], x_diff[:, i])

            C = (1 - c1 - cmu) * C + c1 * np.outer(p_c, p_c) + cmu * rank_mu_update

            # Ensure symmetry (numerical)
            C = (C + C.T) / 2

            # Step size update
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / (np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))) - 1))

            # Simple safeguard against numeric breakdown
            sigma = max(sigma, 1e-20)
            # Enlarge very small eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-20:
                C += 1e-10 * np.eye(dim)

            # Check for stagnation (optional restart trigger, not implemented)

        return best_x, best_y
