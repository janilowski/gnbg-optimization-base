# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Evolution Strategy generating antithetic mirrored offspring pairs to reduce variance in gradient estimation.
# Search state: Retains distribution mean vector, isotropic step size parameter, and global optimum across generations.
# Candidate generation: Generates symmetric offspring pairs (mu + sigma*z, mu - sigma*z) via isotropic Gaussian perturbations.
# Selection and replacement: Selects the top mu offspring based on objective fitness to compute the new weighted recombination mean.
# Adaptation: Employs a simplified success rule based on the net displacement vector norm to dynamically scale step size.
# Exploration mechanisms: Symmetric mirrored sampling forces multi-directional exploration and prevents directional sampling bias.
# Exploitation mechanisms: Logarithmic rank weighting heavily favors top elite offspring during intermediate recombination.
# Boundary handling: All mirrored offspring candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates offspring in precise mirrored pairs sequentially while checking remaining evaluation budget.
# Closest known influences: Mirrored Sampling in Evolution Strategies / Mirrored CMA-ES (Brockhoff et al.).
# Novelty or unusual aspects: Guarantees exact zero empirical mean offset in the raw unconstrained mutational sample pool.
# Failure modes: Mirrored steps can waste evaluations if the current mean is positioned right against a strict boundary constraint.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.lam = int(min(self.budget // 5, max(16, 2 * self.dim)))
        if self.lam % 2 != 0:
            self.lam += 1
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

            half_lam = self.lam // 2
            for i in range(half_lam):
                if self.eval_count >= self.budget:
                    break

                z = np.random.normal(0, 1, size=self.dim)
                
                # Positive mirrored step
                idx1 = 2 * i
                cand1 = np.clip(mean_x + sigma * z, lb, ub)
                y1 = float(func(cand1))
                self.eval_count += 1
                offspring_x[idx1] = cand1
                offspring_z[idx1] = z
                fitness[idx1] = y1

                if y1 < best_y:
                    best_y = y1
                    best_x = cand1.copy()

                if self.eval_count >= self.budget:
                    break

                # Negative mirrored step
                idx2 = 2 * i + 1
                cand2 = np.clip(mean_x - sigma * z, lb, ub)
                y2 = float(func(cand2))
                self.eval_count += 1
                offspring_x[idx2] = cand2
                offspring_z[idx2] = -z
                fitness[idx2] = y2

                if y2 < best_y:
                    best_y = y2
                    best_x = cand2.copy()

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
