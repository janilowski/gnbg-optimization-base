# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Cholesky-CMA-ES algorithm updating the lower triangular decomposition matrix directly without matrix square root operations.
# Search state: Retains distribution mean vector, lower-triangular Cholesky factor matrix, isotropic step size parameter, and global optimum.
# Candidate generation: Generates offspring batches via matrix multiplication of the Cholesky factor and standard Gaussian vectors.
# Selection and replacement: Selects top mu offspring based on objective fitness to compute the new weighted recombination mean.
# Adaptation: Directly updates the Cholesky factor matrix using exact multiplicative rank-one updates from the leading step vector.
# Exploration mechanisms: Isotropic Gaussian step generation prior to Cholesky transformation ensures unconstrained initial search density.
# Exploitation mechanisms: Rank-one Cholesky updates align search corridors directly along successful descent trajectories.
# Boundary handling: All sampled offspring candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates offspring batches sequentially in generational iterations while checking remaining evaluation budget.
# Closest known influences: Cholesky CMA-ES / Matrix-update CMA (Suttorp et al.).
# Novelty or unusual aspects: Eliminates numerical eigensystem decompositions entirely by maintaining Cholesky factorization directly.
# Failure modes: Quadratic time and memory complexity O(D^2) per iteration limits scalability to extremely high-dimensional spaces under tight budgets.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.lam = int(min(self.budget // 5, max(16, 4 + int(round(3.0 * math.log(self.dim))))))
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

        mean = lb + 0.5 * domain_range
        sigma = 0.25
        A = np.diag(domain_range.copy())
        min_std = 1e-6

        ranks = np.arange(1, self.mu + 1)
        raw_weights = math.log(self.mu + 0.5) - np.log(ranks)
        weights = raw_weights / np.sum(raw_weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c1 = min(0.5, 2.0 / ((self.dim + 1.3) ** 2 + mu_eff))
        c_sigma = 1.0 / (math.sqrt(self.dim) + 1.0)

        while self.eval_count < self.budget:
            offspring_x = np.zeros((self.lam, self.dim))
            offspring_z = np.zeros((self.lam, self.dim))
            fitness = np.full(self.lam, float("inf"))

            for i in range(self.lam):
                if self.eval_count >= self.budget:
                    break

                z = np.random.normal(0, 1, size=self.dim)
                step = A @ z
                cand = np.clip(mean + sigma * step, lb, ub)

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

            sorted_idx = np.argsort(fitness)
            elite_idx = sorted_idx[:self.mu]
            elites_x = offspring_x[elite_idx]
            elites_z = offspring_z[elite_idx]

            z_mean = np.sum(elites_z * weights[:, np.newaxis], axis=0)
            mean = np.sum(elites_x * weights[:, np.newaxis], axis=0)

            # Step size update
            z_norm = np.linalg.norm(z_mean) / math.sqrt(self.dim)
            sigma = sigma * math.exp(c_sigma * (z_norm - 1.0))
            sigma = max(sigma, min_std)

            # Rank-one Cholesky update using best step z0
            z0 = elites_z[0]
            norm_z0_sq = np.dot(z0, z0)

            if norm_z0_sq > 1e-10:
                factor = math.sqrt(1.0 - c1)
                term = (math.sqrt(1.0 + (c1 / (1.0 - c1)) * norm_z0_sq) - 1.0) / norm_z0_sq
                Az = A @ z0
                A = factor * A + (factor * term) * np.outer(Az, z0)

            # Check if Cholesky factor collapsed or exploded
            if np.max(np.abs(A) * sigma / (domain_range[:, np.newaxis] + 1e-12)) < 1e-5 or np.max(np.abs(A)) > 1e10:
                mean = np.random.uniform(lb, ub, size=self.dim)
                sigma = 0.25
                A = np.diag(domain_range.copy())

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
