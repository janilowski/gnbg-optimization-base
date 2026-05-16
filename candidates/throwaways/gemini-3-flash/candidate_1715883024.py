# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Cellular Genetic Algorithm structuring population candidates along a 1D ring topology with localized neighborhood interactions.
# Search state: Retains population vectors, objective fitness values, and global best solution across generations.
# Candidate generation: Offspring are created via arithmetic recombination between an individual and its fittest topological neighbor, followed by Gaussian mutation.
# Selection and replacement: Offspring replace their corresponding parent in the ring topology if fitness improves or remains equal.
# Adaptation: Employs a fixed Gaussian mutation rate scaled to domain range to maintain local exploration within cellular niches.
# Exploration mechanisms: Localized ring topology slows the propagation of elite genes across the entire population, preserving diverse niches.
# Exploitation mechanisms: Direct recombination with fittest adjacent neighbors drives local convergence within sub-neighborhoods.
# Boundary handling: Offspring candidate positions are clipped inside domain bounds prior to evaluation.
# Budget strategy: Evaluates cellular offspring sequentially across the ring topology while checking remaining evaluation budget.
# Closest known influences: Cellular Genetic Algorithms (Alba & Dorigo).
# Novelty or unusual aspects: Implements a continuous localized ring neighborhood paired with arithmetic average recombination.
# Failure modes: Slow information dissemination across the ring topology can delay convergence on unimodal convex basins.
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

        sigma = 0.1
        mut_prob = 0.2

        while self.eval_count < self.budget:
            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # 1D Ring Topology: neighbors at i-1 and i+1
                left_idx = (i - 1) % self.pop_size
                right_idx = (i + 1) % self.pop_size

                # Choose best neighbor
                neigh_idx = left_idx if fitness[left_idx] < fitness[right_idx] else right_idx

                # Arithmetic Crossover
                alpha = np.random.uniform(0.2, 0.8)
                offspring = alpha * pop[i] + (1.0 - alpha) * pop[neigh_idx]

                # Gaussian Mutation
                if np.random.rand() < mut_prob:
                    step = np.random.normal(0, 1, size=self.dim) * (sigma * domain_range)
                    offspring += step

                offspring = np.clip(offspring, lb, ub)
                y = float(func(offspring))
                self.eval_count += 1

                if y <= fitness[i]:
                    next_pop[i] = offspring
                    next_fit[i] = y
                else:
                    next_pop[i] = pop[i].copy()
                    next_fit[i] = fitness[i]

                if y < best_y:
                    best_y = y
                    best_x = offspring.copy()

            pop = next_pop
            fitness = next_fit

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
