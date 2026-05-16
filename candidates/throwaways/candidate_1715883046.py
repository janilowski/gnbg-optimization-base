# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A chain-structured Estimation of Distribution Algorithm capturing pairwise correlations between adjacent domain variables.
# Search state: Retains coordinate means, variances, pairwise covariances between adjacent dimensions, and global optimum.
# Candidate generation: Generates solutions via Markov chain conditional Gaussian sampling from dimension 0 sequentially through dim-1.
# Selection and replacement: Evaluates population batch and selects the top elite fraction to update empirical chain moments.
# Adaptation: Adjusts means, variances, and adjacent correlation coefficients dynamically via smoothed elite moment estimation.
# Exploration mechanisms: Stochastic conditional sampling variance ensures active exploration across the domain.
# Exploitation mechanisms: Pairwise correlation tracking aligns search steps along diagonal ridge corridors connecting adjacent variables.
# Boundary handling: All conditionally sampled coordinates are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates sample batches sequentially while strictly checking remaining evaluation budget limits.
# Closest known influences: Estimation of Distribution Algorithms / MIMIC continuous (De Bonet et al.).
# Novelty or unusual aspects: Employs a linear Markov chain correlation structure to achieve O(D) time and memory complexity instead of O(D^2) full covariance matrices.
# Failure modes: Disregards long-range correlations between non-adjacent variables, which can reduce efficiency on fully rotated general landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 6, max(20, 2 * self.dim)))
        if self.pop_size > 60:
            self.pop_size = 60
        self.elite_size = max(3, int(0.3 * self.pop_size))

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
        std = 0.25 * domain_range
        cov_adj = np.zeros(self.dim - 1)  # Covariance between i and i+1
        min_std = 1e-6 * domain_range

        alpha = 0.5  # Smoothing factor

        while self.eval_count < self.budget:
            samples = np.zeros((self.pop_size, self.dim))
            fitness = np.full(self.pop_size, float("inf"))

            # Compute correlation coefficients rho
            rho = np.zeros(self.dim - 1)
            for j in range(self.dim - 1):
                denom = std[j] * std[j + 1]
                if denom > 1e-12:
                    rho[j] = np.clip(cov_adj[j] / denom, -0.95, 0.95)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                cand = np.zeros(self.dim)
                
                # Sample x_0
                z0 = np.random.normal(0, 1)
                cand[0] = np.clip(mean[0] + z0 * std[0], lb[0], ub[0])

                # Sequentially sample x_{j+1} conditioned on x_j
                for j in range(self.dim - 1):
                    cond_mean = mean[j + 1] + rho[j] * (std[j + 1] / (std[j] + 1e-12)) * (cand[j] - mean[j])
                    cond_std = std[j + 1] * np.sqrt(1.0 - rho[j] ** 2)
                    
                    z = np.random.normal(0, 1)
                    cand[j + 1] = np.clip(cond_mean + z * cond_std, lb[j + 1], ub[j + 1])

                samples[i] = cand
                y = float(func(cand))
                self.eval_count += 1
                fitness[i] = y

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            if self.eval_count >= self.budget:
                break

            # Select elites
            sorted_idx = np.argsort(fitness)
            elites = samples[sorted_idx[:self.elite_size]]

            emp_mean = np.mean(elites, axis=0)
            emp_std = np.maximum(np.std(elites, axis=0), min_std)

            emp_cov = np.zeros(self.dim - 1)
            for j in range(self.dim - 1):
                emp_cov[j] = np.mean((elites[:, j] - emp_mean[j]) * (elites[:, j + 1] - emp_mean[j + 1]))

            mean = (1.0 - alpha) * mean + alpha * emp_mean
            std = (1.0 - alpha) * std + alpha * emp_std
            cov_adj = (1.0 - alpha) * cov_adj + alpha * emp_cov

            # Check if collapsed
            if np.max(std / domain_range) < 1e-5:
                mean = np.random.uniform(lb, ub, size=self.dim)
                std = 0.25 * domain_range
                cov_adj = np.zeros(self.dim - 1)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
