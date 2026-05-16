# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Univariate Marginal Distribution Algorithm (UMDA) that models candidate distributions using separable Gaussian probabilities.
# Search state: Retains the current multivariate normal distribution mean vector, diagonal standard deviation vector, and global best.
# Candidate generation: Generates new population batches by independent coordinate sampling from the current Gaussian distribution model.
# Selection and replacement: Selects the top 50% elite population individuals to estimate statistical moments for the next generation.
# Adaptation: Distribution mean shifts towards elite center of mass; standard deviation contracts around converging valleys.
# Exploration mechanisms: Additive variance smoothing prevents premature variance collapse, ensuring continual local exploration.
# Exploitation mechanisms: Concentrating probability density around elite samples rapidly refines the incumbent solution.
# Boundary handling: All sampled individuals are explicitly clipped to stay within variable bounds.
# Budget strategy: Generates batches of population solutions per generation while strictly monitoring evaluation limits.
# Closest known influences: Univariate Marginal Distribution Algorithm / PBIL in continuous domains (Muehlenbein).
# Novelty or unusual aspects: Combines exponential smoothing of distribution parameters with strict lower variance bounding for robustness.
# Failure modes: Disregards coordinate correlations, making search inefficient on non-separable rotated ridges.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 5, max(16, 2 * self.dim)))
        if self.pop_size > 60:
            self.pop_size = 60
        self.elite_size = max(2, self.pop_size // 2)

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
        std_dev = 0.25 * domain_range
        min_std = 1e-6 * domain_range

        while self.eval_count < self.budget:
            pop = np.zeros((self.pop_size, self.dim))
            fitness = np.full(self.pop_size, float("inf"))

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                z = np.random.normal(0, 1, size=self.dim)
                cand = np.clip(mean + z * std_dev, lb, ub)
                pop[i] = cand

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
            elites = pop[sorted_idx[:self.elite_size]]

            # Estimate parameters
            elite_mean = np.mean(elites, axis=0)
            elite_std = np.std(elites, axis=0)

            elite_std = np.maximum(elite_std, min_std)

            # Smooth update
            mean = 0.2 * mean + 0.8 * elite_mean
            std_dev = 0.2 * std_dev + 0.8 * elite_std

            # Check if converged
            if np.max(std_dev / domain_range) < 1e-5:
                mean = np.random.uniform(lb, ub, size=self.dim)
                std_dev = 0.25 * domain_range

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
