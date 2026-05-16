# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A BIPOP-CMA-ES inspired optimization algorithm interleaving large population restarts and small population regimes.
# Search state: Retains current population regime parameters (large vs small), distribution mean, diagonal variances, step size, and global optimum.
# Candidate generation: Generates offspring batches via independent Gaussian sampling scaled by diagonal coordinate variances and global step size.
# Selection and replacement: Evaluates offspring batches and selects top mu individuals to compute the new weighted recombination mean.
# Adaptation: Automatically restarts and alternates between large population doubling regimes (global search) and stochastic small population regimes (local search).
# Exploration mechanisms: Large population regimes and stochastic restarts maintain robust global exploration across multimodal domains.
# Exploitation mechanisms: Small population regimes focus remaining evaluation budget on rapid local convergence within promising basins.
# Boundary handling: All sampled offspring candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Allocates evaluation budgets dynamically between large and small restart regimes until total budget exhaustion.
# Closest known influences: BIPOP-CMA-ES (Hansen).
# Novelty or unusual aspects: Implements the exact BIPOP regime interleaving architecture within a compact diagonal coordinate framework.
# Failure modes: Can exhaust evaluation budgets rapidly during large population doubling runs if initial regimes stagnate.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0

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

        lam0 = int(min(self.budget // 10, max(12, 4 + int(round(3.0 * math.log(self.dim))))))
        lam_large = lam0
        lam_small = lam0
        evals_large = 0
        evals_small = 0

        min_var = (1e-6 * domain_range) ** 2
        regime_is_large = True

        while self.eval_count < self.budget:
            # Setup regime parameters
            lam = lam_large if regime_is_large else lam_small
            if lam > 100:
                lam = 100
            mu = max(2, lam // 2)

            mean_x = np.random.uniform(lb, ub, size=self.dim)
            if best_x is not None and not regime_is_large and np.random.rand() < 0.5:
                mean_x = best_x.copy()

            sigma = 0.25
            var_vec = (0.5 * domain_range) ** 2

            ranks = np.arange(1, mu + 1)
            raw_weights = math.log(mu + 0.5) - np.log(ranks)
            weights = raw_weights / np.sum(raw_weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_cov = min(1.0, (2.0 * mu_eff - 1.0) / (self.dim + 2.0 * mu_eff + 10.0))
            c_sigma = (mu_eff + 2.0) / (self.dim + mu_eff + 5.0)
            d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (self.dim + 1.0)) - 1.0) + c_sigma
            chi_D = math.sqrt(self.dim) * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * (self.dim ** 2)))

            p_sigma = np.zeros(self.dim)
            stagnation = 0
            best_regime_y = float("inf")
            max_regime_iters = max(10, 500 // lam)

            regime_start_evals = self.eval_count
            iters = 0

            while self.eval_count < self.budget and iters < max_regime_iters:
                iters += 1
                offspring_x = np.zeros((lam, self.dim))
                offspring_z = np.zeros((lam, self.dim))
                fitness = np.full(lam, float("inf"))
                std = np.sqrt(var_vec)

                for i in range(lam):
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
                elite_idx = sorted_idx[:mu]
                elites_x = offspring_x[elite_idx]
                elites_z = offspring_z[elite_idx]

                min_batch_y = fitness[sorted_idx[0]]
                if min_batch_y < best_regime_y:
                    best_regime_y = min_batch_y
                    stagnation = 0
                else:
                    stagnation += 1

                if stagnation > 20:
                    break

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
                    break

            regime_evals = self.eval_count - regime_start_evals
            if regime_is_large:
                evals_large += regime_evals
                lam_large = min(lam_large * 2, 200)
            else:
                evals_small += regime_evals

            # Switch regime based on budget spent
            if evals_large <= evals_small:
                regime_is_large = True
            else:
                regime_is_large = False
                ratio = lam_large / (2.0 * lam0 + 1e-12)
                power = (np.random.rand() ** 2)
                lam_small = max(lam0, int(round(lam0 * (ratio ** power))))

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
