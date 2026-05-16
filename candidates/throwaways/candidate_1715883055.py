# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Evolution Strategy generating mutually orthogonal mutation vectors via QR decomposition to ensure uniform directional exploration.
# Search state: Stores distribution mean vector, isotropic step size parameter, and global optimum across generational iterations.
# Candidate generation: Proposes offspring batches via mutually orthogonal Gaussian step vectors computed using QR matrix decomposition.
# Selection and replacement: Selects top mu offspring based on objective fitness to compute the new weighted recombination mean.
# Adaptation: Global step size expands or contracts dynamically based on the normalized magnitude of the weighted elite step vector.
# Exploration mechanisms: Orthogonal step construction eliminates clustering or redundancy in mutational search directions.
# Exploitation mechanisms: Logarithmic rank weighting heavily favors top elite offspring during intermediate recombination.
# Boundary handling: All orthogonal offspring candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates orthogonal offspring batches sequentially while strictly checking remaining evaluation budget.
# Closest known influences: Orthogonal Evolution Strategies / Orthogonal CMA-ES (Choromanski et al.).
# Novelty or unusual aspects: Employs QR decomposition on random Gaussian matrices to enforce exact orthogonality across arbitrary dimension sub-spaces.
# Failure modes: QR decomposition overhead can become computationally expensive for extremely large population batches in high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.lam = int(min(self.budget // 5, max(16, 2 * self.dim)))
        if self.lam > 60:
            self.lam = 60
        self.mu = max(2, self.lam // 2)

    def __call__(self, func):
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        self.eval_count = 0
        domain_range = ub - lb

        best_x = None
        best_y = float("inf")

        mean_x = lb + 0.5 * domain_range
        sigma = 0.25 * math.sqrt(np.sum(domain_range ** 2) / self.dim)
        min_sigma = 1e-6 * math.sqrt(np.sum(domain_range ** 2) / self.dim)

        ranks = np.arange(1, self.mu + 1)
        raw_weights = math.log(self.mu + 0.5) - np.log(ranks)
        weights = raw_weights / np.sum(raw_weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = 1.0 / (math.sqrt(mu_eff) + 2.0)

        while self.eval_count < self.budget:
            offspring_x = np.zeros((self.lam, self.dim))
            offspring_z = np.zeros((self.lam, self.dim))
            fitness = np.full(self.lam, float("inf"))

            # Generate orthogonal step directions using QR decomposition
            n_orth = min(self.lam, self.dim)
            Z = np.random.normal(0, 1, size=(n_orth, self.dim))
            
            if n_orth <= self.dim:
                Q, R = np.linalg.qr(Z.T)
                Q = Q.T  # Shape: (n_orth, dim), rows are orthogonal unit vectors
            else:
                Q = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

            # Scale so expected Euclidean norm matches standard normal vectors (sqrt(dim))
            norm_scale = math.sqrt(self.dim)
            orth_z = Q * norm_scale

            for i in range(self.lam):
                if self.eval_count >= self.budget:
                    break

                if i < n_orth:
                    z = orth_z[i]
                else:
                    z = np.random.normal(0, 1, size=self.dim)

                cand = np.clip(mean_x + sigma * z, lb, ub)
                y = float(func(cand))
                self.eval_count += 1

                offspring_x[i] = cand
                offspring_z[i] = z
                fitness[i] = y

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            if self.eval_count >= self.budget:
                break

            sorted_idx = np.argsort(fitness)[:self.mu]
            elites_x = offspring_x[sorted_idx]
            elites_z = offspring_z[sorted_idx]

            mean_x = np.sum(elites_x * weights[:, np.newaxis], axis=0)
            z_mean = np.sum(elites_z * weights[:, np.newaxis], axis=0)

            z_norm = np.linalg.norm(z_mean) / math.sqrt(self.dim / mu_eff + 1e-12)
            sigma = sigma * math.exp(c_sigma * (z_norm - 1.0))
            sigma = max(sigma, min_sigma)

            if sigma < min_sigma * 10:
                mean_x = np.random.uniform(lb, ub, size=self.dim)
                sigma = 0.25 * math.sqrt(np.sum(domain_range ** 2) / self.dim)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
