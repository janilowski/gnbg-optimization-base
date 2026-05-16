# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Cross-Entropy Method (CEM) minimizing Kullback-Leibler divergence towards elite sample distributions.
# Search state: Retains distribution mean vector, diagonal variance vector, and global best solution across iterations.
# Candidate generation: Generates sample batches via independent coordinate draws from a multivariate diagonal Gaussian distribution.
# Selection and replacement: Identifies the top quantile (elite fraction) from the evaluated batch to compute empirical target moments.
# Adaptation: Updates mean and variance vectors via exponential smoothing towards empirical elite sample moments.
# Exploration mechanisms: Additive minimum variance lower bounds prevent premature collapse of search density.
# Exploitation mechanisms: Concentrating the Gaussian sampling distribution around the top 20% elite quantile rapidly refines promising regions.
# Boundary handling: All sampled candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates sample batches sequentially in generational iterations while checking remaining evaluation budget.
# Closest known influences: Cross-Entropy Method CEM (Rubinstein & Kroese).
# Novelty or unusual aspects: Employs smoothed variance updates combined with exact lower bound constraints to maintain numerical stability.
# Failure modes: Diagonal covariance assumption can result in slower convergence along rotated non-separable landscape ridges.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.batch_size = int(min(self.budget // 5, max(20, 2 * self.dim)))
        if self.batch_size > 60:
            self.batch_size = 60
        self.elite_size = max(2, int(0.2 * self.batch_size))

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
        var = (0.25 * domain_range) ** 2
        min_var = (1e-6 * domain_range) ** 2

        alpha = 0.7  # Smoothing for mean
        beta = 0.7   # Smoothing for variance

        while self.eval_count < self.budget:
            samples = np.zeros((self.batch_size, self.dim))
            fitness = np.full(self.batch_size, float("inf"))
            std = np.sqrt(var)

            for i in range(self.batch_size):
                if self.eval_count >= self.budget:
                    break
                z = np.random.normal(0, 1, size=self.dim)
                cand = np.clip(mean + z * std, lb, ub)
                samples[i] = cand

                y = float(func(cand))
                self.eval_count += 1
                fitness[i] = y

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            if self.eval_count >= self.budget:
                break

            # Select elite fraction
            sorted_idx = np.argsort(fitness)
            elites = samples[sorted_idx[:self.elite_size]]

            # Empirical moments
            elite_mean = np.mean(elites, axis=0)
            elite_var = np.var(elites, axis=0)

            # Smooth updates
            mean = (1.0 - alpha) * mean + alpha * elite_mean
            var = (1.0 - beta) * var + beta * elite_var
            var = np.maximum(var, min_var)

            # Check for collapse
            if np.max(np.sqrt(var) / domain_range) < 1e-5:
                mean = np.random.uniform(lb, ub, size=self.dim)
                var = (0.25 * domain_range) ** 2

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
