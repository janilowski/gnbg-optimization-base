# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Two-Point Step Size Adaptation Evolution Strategy (TPA-ES) probing symmetric test points along the previous step trajectory.
# Search state: Retains distribution mean vector, previous mean vector, isotropic step size parameter, and global optimum.
# Candidate generation: Generates two symmetric test points along the previous step direction, followed by isotropic Gaussian offspring sampling.
# Selection and replacement: Evaluates offspring batch and selects top mu individuals to compute the new weighted recombination mean.
# Adaptation: Global step size expands or contracts dynamically based on whether the forward test point outperforms the backward test point.
# Exploration mechanisms: Isotropic Gaussian offspring sampling ensures multi-directional exploration around the current mean.
# Exploitation mechanisms: Directional test probes specifically accelerate step sizes when progressing smoothly down linear slope corridors.
# Boundary handling: All test probes and offspring candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Allocates evaluations to test probes and offspring batches sequentially while checking remaining evaluation budget.
# Closest known influences: Two-Point Step Size Adaptation TPA-ES (Hansen et al.).
# Novelty or unusual aspects: Directly embeds directional trajectory probing into the generation loop without accumulating separate historical path variables.
# Failure modes: Symmetric directional probing can oscillate in tight curved valleys where forward steps repeatedly hit boundary constraints.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.lam = int(min(self.budget // 5, max(16, 4 + int(round(3.0 * math.log(self.dim))))))
        if self.lam < 4:
            self.lam = 4
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
        prev_mean = mean_x.copy()
        sigma = 0.25 * math.sqrt(np.sum(domain_range ** 2) / self.dim)
        min_sigma = 1e-6 * math.sqrt(np.sum(domain_range ** 2) / self.dim)

        ranks = np.arange(1, self.mu + 1)
        raw_weights = math.log(self.mu + 0.5) - np.log(ranks)
        weights = raw_weights / np.sum(raw_weights)

        alpha_tpa = 0.5 * math.sqrt(self.dim)
        beta_tpa = 0.2

        while self.eval_count < self.budget:
            offspring_x = np.zeros((self.lam, self.dim))
            fitness = np.full(self.lam, float("inf"))

            # Calculate step direction
            diff = mean_x - prev_mean
            norm_diff = np.linalg.norm(diff)

            start_idx = 0

            if norm_diff > 1e-12 and self.eval_count < self.budget - 1:
                v = diff / norm_diff
                t1 = np.clip(mean_x + alpha_tpa * sigma * v, lb, ub)
                t2 = np.clip(mean_x - alpha_tpa * sigma * v, lb, ub)

                y1 = float(func(t1))
                self.eval_count += 1
                offspring_x[0] = t1
                fitness[0] = y1

                if y1 < best_y:
                    best_y = y1
                    best_x = t1.copy()

                y2 = float(func(t2))
                self.eval_count += 1
                offspring_x[1] = t2
                fitness[1] = y2

                if y2 < best_y:
                    best_y = y2
                    best_x = t2.copy()

                # TPA update
                if y1 < y2:
                    sigma *= math.exp(beta_tpa)
                else:
                    sigma *= math.exp(-beta_tpa)

                start_idx = 2

            for i in range(start_idx, self.lam):
                if self.eval_count >= self.budget:
                    break

                z = np.random.normal(0, 1, size=self.dim)
                cand = np.clip(mean_x + sigma * z, lb, ub)

                y = float(func(cand))
                self.eval_count += 1
                offspring_x[i] = cand
                fitness[i] = y

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            if self.eval_count >= self.budget:
                break

            sorted_idx = np.argsort(fitness)[:self.mu]
            elites_x = offspring_x[sorted_idx]

            prev_mean = mean_x.copy()
            mean_x = np.sum(elites_x * weights[:, np.newaxis], axis=0)
            sigma = max(sigma, min_sigma)

            if sigma < min_sigma * 10:
                mean_x = np.random.uniform(lb, ub, size=self.dim)
                prev_mean = mean_x.copy()
                sigma = 0.25 * math.sqrt(np.sum(domain_range ** 2) / self.dim)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
