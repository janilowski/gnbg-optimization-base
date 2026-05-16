# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A hybrid optimization algorithm combining Covariance Matrix Adaptation Evolution Strategies with Differential Evolution recombination operators.
# Search state: Retains distribution mean, diagonal variances, step size parameter, elite parent archive, and global optimum.
# Candidate generation: Half the offspring batch is generated via diagonal Gaussian sampling; half is generated via Differential Evolution mutation on elite parents.
# Selection and replacement: Selects top mu offspring across both proposal regimes to compute the new weighted distribution mean and update variances.
# Adaptation: Adapts diagonal coordinate variances via rank-mu updates and dynamically adjusts global step size based on cumulative path length.
# Exploration mechanisms: Differential Evolution difference vectors sampled across elite parents maintain mutational diversity and prevent premature stagnation.
# Exploitation mechanisms: Weighted mean recombination and variance contraction aggressively pull the search distribution into optimal basins.
# Boundary handling: All sampled and recombined candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates hybrid offspring batches sequentially while strictly monitoring remaining evaluation budget limits.
# Closest known influences: Hybrid CMA-ES / Differential Evolution.
# Novelty or unusual aspects: Directly pits Gaussian sampling against difference vector recombination within a unified competitive selection pool.
# Failure modes: Can experience scaling discrepancies between Gaussian and DE steps if initial step size parameters are mismatched.
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

        mean_x = lb + 0.5 * domain_range
        sigma = 0.25
        var_vec = (0.5 * domain_range) ** 2
        min_var = (1e-6 * domain_range) ** 2

        ranks = np.arange(1, self.mu + 1)
        raw_weights = math.log(self.mu + 0.5) - np.log(ranks)
        weights = raw_weights / np.sum(raw_weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_cov = min(1.0, (2.0 * mu_eff - 1.0) / (self.dim + 2.0 * mu_eff + 10.0))
        c_sigma = (mu_eff + 2.0) / (self.dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (self.dim + 1.0)) - 1.0) + c_sigma
        chi_D = math.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * (self.dim ** 2)))

        p_sigma = np.zeros(self.dim)

        # Archive of elite parents for DE
        elite_parents = np.random.uniform(lb, ub, size=(self.mu, self.dim))
        half_lam = self.lam // 2

        f, cr = 0.8, 0.8

        while self.eval_count < self.budget:
            offspring_x = np.zeros((self.lam, self.dim))
            offspring_z = np.zeros((self.lam, self.dim))
            fitness = np.full(self.lam, float("inf"))
            std = np.sqrt(var_vec)

            for i in range(self.lam):
                if self.eval_count >= self.budget:
                    break

                if i < half_lam:
                    # CMA Gaussian sampling
                    z = np.random.normal(0, 1, size=self.dim)
                    cand = np.clip(mean_x + sigma * std * z, lb, ub)
                else:
                    # DE on elite_parents
                    r1 = np.random.randint(self.mu)
                    r2 = np.random.randint(self.mu)
                    while r2 == r1:
                        r2 = np.random.randint(self.mu)
                    r3 = np.random.randint(self.mu)
                    while r3 == r1 or r3 == r2:
                        r3 = np.random.randint(self.mu)

                    v = elite_parents[r1] + f * (elite_parents[r2] - elite_parents[r3])
                    mask = np.random.rand(self.dim) <= cr
                    mask[np.random.randint(self.dim)] = True
                    target_parent = elite_parents[i % self.mu]
                    cand = np.clip(np.where(mask, v, target_parent), lb, ub)
                    
                    # Estimate z vector for covariance update
                    z = (cand - mean_x) / (sigma * std + 1e-12)

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
            elite_parents = elites_x.copy()

            z_mean = np.sum(elites_z * weights[:, np.newaxis], axis=0)
            mean_x = np.sum(elites_x * weights[:, np.newaxis], axis=0)

            p_sigma = (1.0 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * z_mean
            norm_p = np.linalg.norm(p_sigma)
            sigma = sigma * math.exp((c_sigma / d_sigma) * (norm_p / chi_D - 1.0))
            sigma = max(sigma, 1e-5)

            pos_update = np.sum((elites_z ** 2) * weights[:, np.newaxis], axis=0)
            var_vec = (1.0 - c_cov) * var_vec + c_cov * (var_vec * pos_update)
            var_vec = np.maximum(var_vec, min_var)

            if np.max(np.sqrt(var_vec) * sigma / domain_range) < 1e-5:
                mean_x = np.random.uniform(lb, ub, size=self.dim)
                sigma = 0.25
                var_vec = (0.5 * domain_range) ** 2
                p_sigma = np.zeros(self.dim)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
