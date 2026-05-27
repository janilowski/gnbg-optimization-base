# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An Active Covariance Matrix Adaptation algorithm (sep-CMA-ES) incorporating negative variance updates from poorly performing offspring.
# Search state: Retains distribution mean vector, diagonal variance vector, global step size parameter, evolution paths, and global optimum.
# Candidate generation: Generates offspring batches via independent Gaussian sampling scaled by diagonal coordinate variances and global step size.
# Selection and replacement: Evaluates offspring batch, selecting top mu individuals for positive updates and worst mu_neg individuals for negative variance updates.
# Adaptation: Updates diagonal variances by adding positive rank-mu updates from elite steps and subtracting scaled updates from worst descent directions.
# Exploration mechanisms: Negative variance updates actively deflate search variance along deceptive or failing landscape corridors.
# Exploitation mechanisms: Positive rank-mu variance updates and weighted mean recombination aggressively focus search in promising basins.
# Boundary handling: All sampled offspring candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates offspring batches sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Active CMA-ES (Jastrebski & Arnold).
# Novelty or unusual aspects: Implements negative active variance updates in a diagonal coordinate space without requiring matrix inversions.
# Failure modes: Can experience numerical variance instability if negative update learning rates are set too aggressively.
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
        self.mu_neg = max(1, self.lam // 4)

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

        # Positive weights
        ranks = np.arange(1, self.mu + 1)
        raw_weights = math.log(self.mu + 0.5) - np.log(ranks)
        weights = raw_weights / np.sum(raw_weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Negative weights
        ranks_neg = np.arange(1, self.mu_neg + 1)
        raw_neg = math.log(self.mu_neg + 0.5) - np.log(ranks_neg)
        weights_neg = -raw_neg / np.sum(raw_neg)  # Sums to -1

        c_cov = min(1.0, (2.0 * mu_eff - 1.0) / (self.dim + 2.0 * mu_eff + 10.0))
        c_neg = 0.5 * c_cov  # Dampened negative learning rate

        c_sigma = (mu_eff + 2.0) / (self.dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (self.dim + 1.0)) - 1.0) + c_sigma
        chi_D = math.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * (self.dim ** 2)))

        p_sigma = np.zeros(self.dim)

        while self.eval_count < self.budget:
            offspring_x = np.zeros((self.lam, self.dim))
            offspring_z = np.zeros((self.lam, self.dim))
            fitness = np.full(self.lam, float("inf"))

            std = np.sqrt(var_vec)

            for i in range(self.lam):
                if self.eval_count >= self.budget:
                    break

                z = np.random.normal(0, 1, size=self.dim)
                cand = np.clip(mean_x + sigma * std * z, lb, ub)

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
            worst_idx = sorted_idx[-self.mu_neg:]

            elites_x = offspring_x[elite_idx]
            elites_z = offspring_z[elite_idx]
            worst_z = offspring_z[worst_idx]

            z_mean = np.sum(elites_z * weights[:, np.newaxis], axis=0)
            mean_x = np.sum(elites_x * weights[:, np.newaxis], axis=0)

            # Cumulative path for sigma
            p_sigma = (1.0 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * z_mean
            norm_p = np.linalg.norm(p_sigma)
            sigma = sigma * math.exp((c_sigma / d_sigma) * (norm_p / chi_D - 1.0))
            sigma = max(sigma, 1e-5)

            # Active variance update (positive + negative)
            pos_update = np.sum((elites_z ** 2) * weights[:, np.newaxis], axis=0)
            neg_update = np.sum((worst_z ** 2) * weights_neg[:, np.newaxis], axis=0)  # weights_neg are negative

            var_vec = (1.0 - c_cov) * var_vec + c_cov * (var_vec * pos_update) + c_neg * (var_vec * neg_update)
            var_vec = np.maximum(var_vec, min_var)

            if np.max(np.sqrt(var_vec) * sigma / domain_range) < 1e-5:
                mean_x = np.random.uniform(lb, ub, size=self.dim)
                sigma = 0.25
                var_vec = (0.5 * domain_range) ** 2
                p_sigma = np.zeros(self.dim)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
