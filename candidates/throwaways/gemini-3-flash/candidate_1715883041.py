# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Population-Based Incremental Learning (PBIL) algorithm estimating independent Gaussian coordinate distributions.
# Search state: Retains current mean vector, diagonal standard deviation vector, and global best solution found.
# Candidate generation: Generates batches of population candidates via independent coordinate sampling from the current Gaussian probability model.
# Selection and replacement: Selects the top elite subgroup from the evaluated batch to calculate updated statistical target moments.
# Adaptation: Exponentially smooths the distribution mean and standard deviation towards the elite sample moments each generation.
# Exploration mechanisms: Additive variance smoothing lower bounds ensure continuous stochastic exploration around the converging mean.
# Exploitation mechanisms: Shifting the distribution mean towards elite individuals rapidly concentrates probability mass around successful basins.
# Boundary handling: All sampled candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates population samples in generational batches while strictly adhering to evaluation budget ceilings.
# Closest known influences: Population-Based Incremental Learning PBIL (Baluja).
# Novelty or unusual aspects: Employs independent variance updates per dimension with robust lower bound clipping to prevent numerical degeneration.
# Failure modes: Disregards variable correlations, leading to potential inefficiency along highly non-separable diagonal valleys.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 6, max(16, 2 * self.dim)))
        if self.pop_size > 50:
            self.pop_size = 50

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

        alpha = 0.25  # Learning rate for mean
        beta = 0.15   # Learning rate for std
        elite_size = max(2, self.pop_size // 4)

        while self.eval_count < self.budget:
            pop = np.zeros((self.pop_size, self.dim))
            fitness = np.full(self.pop_size, float("inf"))

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                step = np.random.normal(0, 1, size=self.dim) * std_dev
                cand = np.clip(mean + step, lb, ub)

                y = float(func(cand))
                self.eval_count += 1
                pop[i] = cand
                fitness[i] = y

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            if self.eval_count >= self.budget:
                break

            # Select elite subgroup
            sorted_idx = np.argsort(fitness)
            elites = pop[sorted_idx[:elite_size]]

            # Estimate target moments
            elite_mean = np.mean(elites, axis=0)
            elite_std = np.sqrt(np.mean((elites - mean) ** 2, axis=0))

            # Update distribution moments via exponential smoothing
            mean = (1.0 - alpha) * mean + alpha * elite_mean
            std_dev = (1.0 - beta) * std_dev + beta * elite_std
            std_dev = np.maximum(std_dev, min_std)

            # Check if variance collapsed
            if np.max(std_dev / domain_range) < 1e-5:
                mean = np.random.uniform(lb, ub, size=self.dim)
                std_dev = 0.25 * domain_range

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
