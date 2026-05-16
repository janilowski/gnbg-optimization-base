# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An Elite Subspace Differential Evolution algorithm performing singular value decomposition on elite archives to guide mutational steps.
# Search state: Retains active population positions, objective fitness values, and global optimum across generational iterations.
# Candidate generation: Alternates between standard DE mutation and Gaussian perturbations directed along the principal component subspace of elite solutions.
# Selection and replacement: Standard one-to-one parent replacement; re-sorts swarm each generation to extract top elite solution matrices.
# Adaptation: Principal component basis vectors automatically align mutational steps along the dominant orientation of local fitness valleys.
# Exploration mechanisms: Standard DE/best/1/bin difference vectors maintain robust mutational reach across all coordinate dimensions.
# Exploitation mechanisms: Subspace Gaussian mutations aggressively focus search along the exact principal ridges of the elite optimum basin.
# Boundary handling: All trial vector positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population trial vectors sequentially in generational iterations while checking remaining evaluation budget.
# Closest known influences: Subspace Differential Evolution / Eigen DE / SVD-Assisted Evolutionary Algorithms.
# Novelty or unusual aspects: Directly embeds exact singular value decomposition (SVD) on elite subsets to construct orthogonal mutation bases.
# Failure modes: SVD matrix operations can become computationally expensive for extremely large elite sample matrices in high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 8, max(25, 2 * self.dim)))
        if self.pop_size > 60:
            self.pop_size = 60
        self.n_elite = max(4, min(10, self.pop_size // 3))

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
            sorted_idx = np.argsort(fitness)
            elite_pop = pop[sorted_idx[:self.n_elite]]
            best_curr_x = elite_pop[0]

            # Compute SVD on elite population
            mu_elite = np.mean(elite_pop, axis=0)
            centered = elite_pop - mu_elite

            try:
                _, s_vals, vh = np.linalg.svd(centered, full_matrices=False)
                # vh shape: (n_elite, dim)
                n_basis = min(self.n_elite, self.dim)
                basis = vh[:n_basis]
                stds = np.maximum(s_vals[:n_basis] / math.sqrt(self.n_elite), 1e-6)
            except np.linalg.LinAlgError:
                basis = np.eye(self.dim)[:min(self.n_elite, self.dim)]
                stds = np.full(len(basis), 0.1)

            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                if np.random.rand() < 0.5:
                    # Standard DE/best/1/bin
                    r1 = np.random.randint(self.pop_size)
                    while r1 == i:
                        r1 = np.random.randint(self.pop_size)

                    r2 = np.random.randint(self.pop_size)
                    while r2 == i or r2 == r1:
                        r2 = np.random.randint(self.pop_size)

                    v = best_curr_x + f * (pop[r1] - pop[r2])
                    mask = np.random.rand(self.dim) <= cr
                    mask[np.random.randint(self.dim)] = True
                    trial = np.where(mask, v, pop[i])
                else:
                    # Elite subspace Gaussian mutation
                    z = np.random.normal(0, 1, size=len(basis))
                    step = np.dot(z * stds, basis)
                    trial = pop[i] + step

                trial = np.clip(trial, lb, ub)
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
