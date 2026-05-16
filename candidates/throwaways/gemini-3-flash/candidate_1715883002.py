# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A lightweight diagonal Covariance Matrix Adaptation Evolution Strategy (sep-CMA-ES) that maintains variance per coordinate axis.
# Search state: Stores the distribution mean vector, coordinate-wise variance vector, step size sigma, and evolution paths.
# Candidate generation: Offspring are sampled from a multivariate normal distribution with diagonal covariance centered at the current mean.
# Selection and replacement: The top mu offspring are selected and weighted logarithmically to update the distribution mean and coordinate variances.
# Adaptation: Adjusts global step size sigma and coordinate variances using cumulation (evolution paths) based on successful steps.
# Exploration mechanisms: Stochastic sampling with global step size adaptation allows exploration of the search space without premature collapse.
# Exploitation mechanisms: Weighted recombination of the best parents shifts the mean directly towards promising regions.
# Boundary handling: Samples are clamped to bounds, and the mean vector is also kept strictly within bounds.
# Budget strategy: Generational loops produce batches of lambda offspring until the evaluation budget is completely exhausted.
# Closest known influences: sep-CMA-ES (Ros & Hansen).
# Novelty or unusual aspects: Omits full covariance matrix rotations to maintain O(dim) time and space complexity suitable for fast benchmarks.
# Failure modes: Cannot efficiently learn highly rotated valleys due to the diagonal covariance restriction.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.lam = int(4 + math.floor(3 * math.log(self.dim)))
        if self.lam > budget // 2:
            self.lam = max(2, budget // 2)
        self.mu = int(math.floor(self.lam / 2))

    def __call__(self, func):
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        self.eval_count = 0
        best_x = None
        best_y = float("inf")

        # Initial mean in the center of bounds
        mean = lb + 0.5 * (ub - lb)
        sigma = 0.25
        diag_C = np.ones(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)

        # Weights
        weights = np.array([math.log(self.mu + 0.5) - math.log(i + 1) for i in range(self.mu)])
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)

        # Learning rates
        cs = (mueff + 2) / (self.dim + mueff + 5)
        damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (self.dim + 1)) - 1) + cs
        cc = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        c1 = 2 / ((self.dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2)**2 + mueff))

        chiN = math.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim**2))
        bound_range = ub - lb

        while self.eval_count < self.budget:
            pop = np.zeros((self.lam, self.dim))
            fitness = np.zeros(self.lam)

            for i in range(self.lam):
                if self.eval_count >= self.budget:
                    fitness[i] = float("inf")
                    continue
                # Sample
                std_dev = sigma * np.sqrt(diag_C) * bound_range
                z = np.random.normal(0, 1, self.dim)
                x = mean + std_dev * z
                x = np.clip(x, lb, ub)
                pop[i] = x

                y = float(func(x))
                self.eval_count += 1
                fitness[i] = y

                if y < best_y:
                    best_y = y
                    best_x = x.copy()

            if self.eval_count >= self.budget:
                break

            # Sort population
            idx = np.argsort(fitness)
            pop_best = pop[idx[:self.mu]]

            # Update mean
            mean_old = mean.copy()
            mean = np.sum(pop_best * weights[:, np.newaxis], axis=0)
            mean = np.clip(mean, lb, ub)

            # Update evolution paths
            y_step = (mean - mean_old) / (sigma * bound_range)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (y_step / np.sqrt(diag_C))
            norm_ps = np.linalg.norm(ps)
            hsig = 1 if norm_ps / math.sqrt(1 - (1 - cs)**(2 * self.eval_count / self.lam)) < (1.4 + 2 / (self.dim + 1)) * chiN else 0
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_step

            # Update diagonal covariance
            artmp = (pop_best - mean_old) / (sigma * bound_range)
            diag_C = (1 - c1 - cmu) * diag_C + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * diag_C)
            diag_C += cmu * np.sum(weights[:, np.newaxis] * artmp**2, axis=0)
            diag_C = np.clip(diag_C, 1e-12, 1e12)

            # Update sigma
            sigma *= np.exp((cs / damps) * (norm_ps / chiN - 1))
            if sigma < 1e-12:
                sigma = 0.25
                diag_C = np.ones(self.dim)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
