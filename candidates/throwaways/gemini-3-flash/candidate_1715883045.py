# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Gaussian Mixture Model Estimation of Distribution Algorithm partitioning candidates into elite and background clusters.
# Search state: Retains statistical means and diagonal variances for two mixture clusters (elite and background), along with global optimum.
# Candidate generation: Offspring are sampled predominantly (80%) from the elite cluster distribution and secondarily (20%) from the background cluster.
# Selection and replacement: Ranks the evaluated population each generation to update the elite (top 30%) and background partition memberships.
# Adaptation: Cluster moments dynamically track empirical sample distributions with exponential smoothing to prevent abrupt variance collapse.
# Exploration mechanisms: Sampling the background cluster maintains wide search trajectories across secondary domain basins.
# Exploitation mechanisms: Dominant sampling of the elite cluster aggressively refines the incumbent solution basin.
# Boundary handling: All sampled candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates generational offspring batches sequentially while rigorously checking remaining evaluation budget.
# Closest known influences: Estimation of Distribution Algorithms with Mixture Models / EEDA (Muehlenbein).
# Novelty or unusual aspects: Lightweight dual-cluster mixture model separating elite exploitation from background exploration without iterative EM clustering.
# Failure modes: Assumes diagonal separability within clusters, which can degrade efficiency on rotated non-separable landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 5, max(20, 2 * self.dim)))
        if self.pop_size > 60:
            self.pop_size = 60
        self.n_elite = max(3, int(0.3 * self.pop_size))
        self.n_bg = self.pop_size - self.n_elite

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

        pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        fitness = np.full(self.pop_size, float("inf"))

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop[i]))
            self.eval_count += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        min_std = 1e-6 * domain_range
        alpha = 0.5  # Smoothing for moment updates

        # Initial cluster moments
        sorted_idx = np.argsort(fitness)
        elites = pop[sorted_idx[:self.n_elite]]
        bg = pop[sorted_idx[self.n_elite:]]

        mu1, std1 = np.mean(elites, axis=0), np.maximum(np.std(elites, axis=0), min_std)
        mu2, std2 = np.mean(bg, axis=0), np.maximum(np.std(bg, axis=0), min_std)

        while self.eval_count < self.budget:
            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            # Elitism: preserve top individual
            best_idx = np.argmin(fitness)
            next_pop[0] = pop[best_idx].copy()
            next_fit[0] = fitness[best_idx]

            n_elite_samples = int(0.8 * (self.pop_size - 1))
            
            for i in range(1, self.pop_size):
                if self.eval_count >= self.budget:
                    break

                z = np.random.normal(0, 1, size=self.dim)
                if i <= n_elite_samples:
                    cand = mu1 + z * std1
                else:
                    cand = mu2 + z * std2

                cand = np.clip(cand, lb, ub)
                y = float(func(cand))
                self.eval_count += 1

                next_pop[i] = cand
                next_fit[i] = y

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            pop = next_pop
            fitness = next_fit

            if self.eval_count >= self.budget:
                break

            # Update clusters
            sorted_idx = np.argsort(fitness)
            elites = pop[sorted_idx[:self.n_elite]]
            bg = pop[sorted_idx[self.n_elite:]]

            emp_mu1 = np.mean(elites, axis=0)
            emp_std1 = np.maximum(np.std(elites, axis=0), min_std)
            emp_mu2 = np.mean(bg, axis=0)
            emp_std2 = np.maximum(np.std(bg, axis=0), min_std)

            mu1 = (1.0 - alpha) * mu1 + alpha * emp_mu1
            std1 = (1.0 - alpha) * std1 + alpha * emp_std1
            mu2 = (1.0 - alpha) * mu2 + alpha * emp_mu2
            std2 = (1.0 - alpha) * std2 + alpha * emp_std2

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
