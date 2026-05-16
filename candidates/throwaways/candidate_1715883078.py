# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Differential Evolution algorithm restricting base vector selection to spatial nearest neighbors to promote localized niche exploration.
# Search state: Retains active population positions, objective fitness values, and global optimum across generations.
# Candidate generation: Generates trial vectors by selecting mutation base anchors from spatial k-nearest neighbors combined with global difference pairs.
# Selection and replacement: Standard one-to-one parent replacement; updates local neighborhood spatial matrices each generation.
# Adaptation: Localized base selection naturally preserves multiple population niches across disconnected landscape valleys.
# Exploration mechanisms: Sampling difference vectors (r2, r3) globally from the entire swarm ensures robust inter-niche leaps.
# Exploitation mechanisms: Selecting base vectors specifically from spatial nearest neighbors concentrates fine-tuning within discovered basins.
# Boundary handling: All trial vector positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population trial vectors sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Neighborhood/Proximity Differential Evolution / Species DE / Crowding DE.
# Novelty or unusual aspects: Directly embeds vectorized Euclidean spatial nearest-neighbor indexing into mutation base selection.
# Failure modes: Distance matrix computations O(N^2) per generation can become computationally expensive for extremely large populations.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 8, max(20, 2 * self.dim)))
        if self.pop_size > 60:
            self.pop_size = 60
        self.k_neigh = max(3, min(7, self.pop_size // 3))

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

        f, cr = 0.8, 0.8

        while self.eval_count < self.budget:
            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            # Pairwise distance matrix
            diff_mat = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
            dist_mat = np.linalg.norm(diff_mat, axis=2)
            np.fill_diagonal(dist_mat, float("inf"))

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # k-nearest neighbors
                sorted_neigh = np.argsort(dist_mat[i])[:self.k_neigh]
                r1 = np.random.choice(sorted_neigh)

                r2 = np.random.randint(self.pop_size)
                while r2 == r1 or r2 == i:
                    r2 = np.random.randint(self.pop_size)

                r3 = np.random.randint(self.pop_size)
                while r3 == r1 or r3 == r2 or r3 == i:
                    r3 = np.random.randint(self.pop_size)

                v = pop[r1] + f * (pop[r2] - pop[r3])
                mask = np.random.rand(self.dim) <= cr
                mask[np.random.randint(self.dim)] = True
                trial = np.clip(np.where(mask, v, pop[i]), lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y <= fitness[i]:
                    next_pop[i] = trial
                    next_fit[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()
                else:
                    next_pop[i] = pop[i]
                    next_fit[i] = fitness[i]

            pop = next_pop
            fitness = next_fit

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
