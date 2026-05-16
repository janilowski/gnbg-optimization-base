# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Exponential Natural Evolution Strategy (xNES) adapting distribution parameters via estimated natural gradients.
# Search state: Stores distribution mean vector, diagonal log-standard deviations vector, and global optimum across generations.
# Candidate generation: Generates offspring batches via independent Gaussian sampling scaled by exponentiated log-standard deviations.
# Selection and replacement: Ranks evaluated offspring batch to assign utility weights summing exactly to zero across the population.
# Adaptation: Updates mean and diagonal log-standard deviations along estimated natural gradient directions using rank utility weights.
# Exploration mechanisms: Unconstrained natural gradient updates on log-standard deviations allow smooth continuous variance expansions.
# Exploitation mechanisms: Rank utility weighting heavily prioritizes the top half of the offspring distribution, driving steady local convergence.
# Boundary handling: All sampled offspring candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates offspring batches sequentially while strictly checking remaining evaluation budget.
# Closest known influences: Exponential Natural Evolution Strategies xNES (Glasmachers et al.).
# Novelty or unusual aspects: Pre-computes exact zero-sum utility weights for stable natural gradient ascent in diagonal coordinate space.
# Failure modes: High learning rates on log-variances can cause numerical overflow or underflow if unconstrained.
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
        # Initialize log-std dev: ln(0.25 * range)
        log_std = np.log(0.25 * domain_range)
        min_log_std = np.log(1e-6 * domain_range)

        # Precompute utilities (summing to 0)
        mu = self.lam // 2
        ranks = np.arange(1, self.lam + 1)
        raw_u = np.maximum(0.0, math.log(mu + 0.5) - np.log(ranks))
        u_norm = raw_u / (np.sum(raw_u) + 1e-12) - (1.0 / self.lam)

        eta_mu = 1.0
        eta_s = (3.0 + math.log(self.dim)) / (5.0 * math.sqrt(self.dim))

        while self.eval_count < self.budget:
            offspring_x = np.zeros((self.lam, self.dim))
            offspring_z = np.zeros((self.lam, self.dim))
            fitness = np.full(self.lam, float("inf"))
            std = np.exp(log_std)

            for i in range(self.lam):
                if self.eval_count >= self.budget:
                    break

                z = np.random.normal(0, 1, size=self.dim)
                cand = np.clip(mean + std * z, lb, ub)

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
            sorted_z = offspring_z[sorted_idx]

            # Natural gradients
            G_mu = np.sum(sorted_z * u_norm[:, np.newaxis], axis=0)
            G_s = np.sum((sorted_z ** 2 - 1.0) * u_norm[:, np.newaxis], axis=0)

            # Parameter updates
            mean = mean + eta_mu * std * G_mu
            log_std = log_std + eta_s * G_s
            log_std = np.maximum(log_std, min_log_std)

            # Check for collapse
            if np.max(np.exp(log_std) / domain_range) < 1e-5:
                mean = np.random.uniform(lb, ub, size=self.dim)
                log_std = np.log(0.25 * domain_range)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
