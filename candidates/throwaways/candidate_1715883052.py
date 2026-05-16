# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Evolution Strategy utilizing cumulative step size adaptation (CSA) via an accumulated evolution path vector.
# Search state: Retains distribution mean, isotropic step size parameter, conjugate evolution path vector, and global optimum.
# Candidate generation: Generates offspring batches via isotropic Gaussian sampling scaled by the global step size parameter.
# Selection and replacement: Selects top mu offspring based on objective fitness to compute the new weighted recombination mean.
# Adaptation: Global step size adapts dynamically based on the Euclidean norm of the accumulated conjugate evolution path relative to expected random walk length.
# Exploration mechanisms: Cumulative path tracking dampens stochastic fluctuations and maintains steady step sizes across linear valleys.
# Exploitation mechanisms: Logarithmic rank weighting heavily favors top elite offspring during intermediate recombination.
# Boundary handling: All sampled candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates offspring batches sequentially in generational iterations while checking remaining evaluation budget.
# Closest known influences: Cumulative Step Size Adaptation CSA-ES (Hansen & Ostermeier).
# Novelty or unusual aspects: Employs exact numerical expectation scaling (chi_D) for stable path accumulation across arbitrary dimensions.
# Failure modes: Isotropic step size assumption limits acceleration on ill-conditioned rotated landscapes.
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
        sigma = 0.25 * math.sqrt(np.sum(domain_range ** 2) / self.dim)
        min_sigma = 1e-6 * math.sqrt(np.sum(domain_range ** 2) / self.dim)

        # Precompute weights
        ranks = np.arange(1, self.mu + 1)
        raw_weights = math.log(self.mu + 0.5) - np.log(ranks)
        weights = raw_weights / np.sum(raw_weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # CSA parameters
        c_sigma = (mu_eff + 2.0) / (self.dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (self.dim + 1.0)) - 1.0) + c_sigma
        chi_D = math.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * (self.dim ** 2)))

        p_sigma = np.zeros(self.dim)

        while self.eval_count < self.budget:
            offspring_x = np.zeros((self.lam, self.dim))
            offspring_z = np.zeros((self.lam, self.dim))
            fitness = np.full(self.lam, float("inf"))

            for i in range(self.lam):
                if self.eval_count >= self.budget:
                    break

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

            # Recombination
            mean_x = np.sum(elites_x * weights[:, np.newaxis], axis=0)
            z_mean = np.sum(elites_z * weights[:, np.newaxis], axis=0)

            # Cumulative path update
            p_sigma = (1.0 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * z_mean
            
            # Step size adaptation
            norm_p = np.linalg.norm(p_sigma)
            sigma = sigma * math.exp((c_sigma / d_sigma) * (norm_p / chi_D - 1.0))
            sigma = max(sigma, min_sigma)

            # Check for collapse
            if sigma < min_sigma * 10:
                mean_x = np.random.uniform(lb, ub, size=self.dim)
                sigma = 0.25 * math.sqrt(np.sum(domain_range ** 2) / self.dim)
                p_sigma = np.zeros(self.dim)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
