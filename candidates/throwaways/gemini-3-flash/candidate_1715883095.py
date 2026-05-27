# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Evolution Strategy augmenting diagonal covariance adaptation with explicit population velocity momentum.
# Search state: Retains distribution mean vector, diagonal variance vector, step size parameter, population velocity vector, and global optimum.
# Candidate generation: Proposes offspring batches via independent Gaussian sampling augmented by inertia from past successful mean displacement vectors.
# Selection and replacement: Selects top mu offspring based on objective fitness to compute the new weighted recombination mean.
# Adaptation: Updates diagonal variances via rank-mu updates and adapts global step size based on cumulative path length normalization.
# Exploration mechanisms: Directional population momentum propels search corridors across flat plateaus and prevents premature stagnation.
# Exploitation mechanisms: Weighted recombination and variance contraction aggressively pull search distribution down discovered valleys.
# Boundary handling: All sampled offspring candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates offspring batches sequentially in generational iterations while checking remaining evaluation budget.
# Closest known influences: CMA-ES with Momentum / Velocity Adaptation Evolution Strategies.
# Novelty or unusual aspects: Directly embeds continuous velocity inertia tracking into diagonal CMA-ES sampling equations.
# Failure modes: Can experience overshooting on narrow quadratic valleys if momentum inertia weights are set too aggressively.
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

        vel = np.zeros(self.dim)
        momentum = 0.5

        ranks = np.arange(1, self.mu + 1)
        raw_weights = math.log(self.mu + 0.5) - np.log(ranks)
        weights = raw_weights / np.sum(raw_weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_cov = min(1.0, (2.0 * mu_eff - 1.0) / (self.dim + 2.0 * mu_eff + 10.0))
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
                cand = np.clip(mean_x + momentum * vel + sigma * std * z, lb, ub)

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
            new_mean_x = np.sum(elites_x * weights[:, np.newaxis], axis=0)

            vel = 0.5 * vel + 0.5 * (new_mean_x - mean_x)
            mean_x = new_mean_x

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
                vel = np.zeros(self.dim)
                p_sigma = np.zeros(self.dim)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
