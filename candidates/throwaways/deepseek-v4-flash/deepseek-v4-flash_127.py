import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a variant of the Covariance Matrix Adaptation
# Evolution Strategy (CMA-ES) for black-box minimization of continuous functions.
# It uses rank-one updates and cumulative step-size adaptation.
# Search state: The algorithm maintains a mean vector (m), a covariance matrix (C),
# a global step-size (sigma), two evolution paths (p_c, p_sigma), and generation counter.
# Candidate generation: At each generation, lambda offspring are sampled from
# m + sigma * N(0, C). Samples are reflected back into the feasible domain.
# Selection and replacement: The mu best individuals (by fitness) are selected and
# the mean is updated as their weighted average. The covariance matrix is updated
# using the selected differences and the evolution paths.
# Adaptation: Step-size sigma is adapted using cumulative path length control (CSA)
# to maintain a desired evolution path length. The covariance matrix adapts the
# shape of the search distribution to the local curvature.
# Exploration mechanisms: The stochastic sampling from a multivariate normal
# distribution and the slow adaptation of sigma and C promote exploration.
# Exploitation mechanisms: The weighted recombination of the best solutions
# focuses the search on promising areas, and the shrinking C exploits learned
# second‑order information.
# Boundary handling: Out-of-bounds coordinates are reflected back into the domain
# (mirroring) to keep the sample distribution unbiased.
# Budget strategy: The total number of function evaluations is capped by the
# provided budget. The algorithm runs generations until the budget is exhausted.
# Closest known influences: Original CMA-ES by Hansen and Ostermeier (2001).
# No restarts, no weighted selection beyond truncation.
# Novelty or unusual aspects: Minimal implementation; uses only rank-one update.
# No diagonal decoding or active update.
# Failure modes: May stagnate if the initial covariance is inappropriate, or if
# the function is extremely multimodal with long narrow valleys.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        # algorithm parameters (tuned for typical use)
        self.lambda_ = 4 + int(3 * np.log(self.dim))
        self.mu = self.lambda_ // 2
        # weights for recombination
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights**2)  # effective population size

        # learning rates
        self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dim + 2)**2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs

        # state variables (initialized in __call__ because bounds influence initial mean)
        self.m = None
        self.sigma = None
        self.C = None
        self.p_c = None
        self.p_sigma = None
        self.evaluations = 0
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float).ravel()
            ub = np.asarray(func.upper, dtype=float).ravel()
        elif hasattr(func, 'bounds'):
            bounds = func.bounds
            lb = np.asarray(bounds.lb, dtype=float).ravel()
            ub = np.asarray(bounds.ub, dtype=float).ravel()
        else:
            raise AttributeError("func must have either lower/upper or bounds.lb/bounds.ub")
        self.dim = len(lb)  # ensure dimensionality matches

        # Reset state
        self.m = (lb + ub) / 2.0
        self.sigma = 0.3 * (ub[0] - lb[0]) if self.dim > 0 else 1.0  # initial step size ~ 30% of range
        self.C = np.eye(self.dim)
        self.p_c = np.zeros(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.evaluations = 0
        self.best_y = np.inf
        self.best_x = None

        # main loop
        while self.evaluations < self.budget:
            # sample offspring
            B = None  # we compute eigendecomposition only if needed for evolution path update later
            # For efficiency, we compute sqrt(C) via Cholesky
            try:
                A = np.linalg.cholesky(self.C)  # C = A A^T
            except np.linalg.LinAlgError:
                # fallback (should not happen often)
                A = np.linalg.cholesky(self.C + 1e-12 * np.eye(self.dim))

            offspring = np.empty((self.lambda_, self.dim))
            fitness = np.empty(self.lambda_)

            # sample and evaluate
            budget_left = self.budget - self.evaluations
            if budget_left < self.lambda_:
                # last incomplete generation
                lam = budget_left
                offspring = np.empty((lam, self.dim))
                fitness = np.empty(lam)
            else:
                lam = self.lambda_

            for i in range(lam):
                z = np.random.randn(self.dim)
                x = self.m + self.sigma * (A @ z)
                # boundary reflection
                x = self._reflect(x, lb, ub)
                offspring[i] = x
                y = func(x)
                self.evaluations += 1
                fitness[i] = y
                if y < self.best_y:
                    self.best_y = y
                    self.best_x = x.copy()

            # selection and recombination
            idx = np.argsort(fitness)
            x_old = self.m.copy()
            x_selected = offspring[idx[:self.mu]]
            # new mean
            self.m = np.dot(self.weights, x_selected)

            # update evolution paths and covariance
            # compute the weighted mean of the selected z's (in original coordinate system)
            # z_i = inv(A) @ (x_i - x_old) / sigma
            A_inv = np.linalg.inv(A)  # small dimension, fine
            zw = np.zeros(self.dim)
            for i in range(self.mu):
                zw += self.weights[i] * (A_inv @ (x_selected[i] - x_old)) / self.sigma

            # update evolution paths
            self.p_c = (1 - self.cc) * self.p_c + np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * zw
            self.p_sigma = (1 - self.cs) * self.p_sigma + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * zw

            # update covariance matrix
            # rank-one update
            self.C = (1 - self.c1 - self.cmu) * self.C \
                     + self.c1 * np.outer(self.p_c, self.p_c)
            # rank-mu update
            for i in range(self.mu):
                z_i = (A_inv @ (x_selected[i] - x_old)) / self.sigma
                self.C += self.cmu * self.weights[i] * np.outer(z_i, z_i)

            # update step size
            sigma_factor = np.exp((self.cs / self.damps) * (np.linalg.norm(self.p_sigma) / np.sqrt(self.dim) - 1))
            self.sigma = self.sigma * sigma_factor

        return self.best_x, self.best_y

    @staticmethod
    def _reflect(x, lb, ub):
        """Reflect coordinate back into [lb, ub] by mirroring."""
        # handle lower bound
        lower_violation = x < lb
        while np.any(lower_violation):
            x[lower_violation] = lb[lower_violation] + (lb[lower_violation] - x[lower_violation])
            lower_violation = x < lb  # handle double reflection
        # handle upper bound
        upper_violation = x > ub
        while np.any(upper_violation):
            x[upper_violation] = ub[upper_violation] - (x[upper_violation] - ub[upper_violation])
            upper_violation = x > ub
        # If still out of bounds (due to numerical issues) clamp:
        x = np.clip(x, lb, ub)
        return x
