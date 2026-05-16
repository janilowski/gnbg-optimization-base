# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Micro-Genetic Algorithm operating on a minimal population with automated diversity checks and random restarts.
# Search state: Retains a tiny population of 6 candidate solution vectors, their objective values, and the global best.
# Candidate generation: Parents chosen via tournament selection generate offspring via uniform crossover without mutational operators.
# Selection and replacement: The top elite individual is preserved, while offspring fill the remaining population slots in each generational cycle.
# Adaptation: Automatically flushes and re-seeds the population with random samples around the elite when population diversity collapses.
# Exploration mechanisms: Frequent random restarts of the non-elite population maintain global exploration across the domain.
# Exploitation mechanisms: Minimal population size and tournament selection force extremely rapid convergence towards local optima.
# Boundary handling: All candidate solutions are constrained via clipping within domain boundaries.
# Budget strategy: Generates micro-population batches sequentially while rigorously checking remaining evaluation budget.
# Closest known influences: Micro-Genetic Algorithm (Krishnakumar).
# Novelty or unusual aspects: Eliminates mutation entirely in favor of frequent automated restarts triggered by Euclidean diversity metrics.
# Failure modes: Can oscillate or stall if deceptive local minima basins are wider than the exploratory restart reach.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = 6  # Minimal population size for micro-GA

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

        stagnation = 0

        while self.eval_count < self.budget:
            # Check diversity
            pop_min = np.min(pop, axis=0)
            pop_max = np.max(pop, axis=0)
            diversity = np.max((pop_max - pop_min) / domain_range)

            if diversity < 1e-4 or stagnation > 15:
                # Restart: keep best at index 0, randomize rest
                best_idx = np.argmin(fitness)
                elite_x = pop[best_idx].copy()
                elite_y = fitness[best_idx]

                pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
                pop[0] = elite_x
                fitness[0] = elite_y

                for i in range(1, self.pop_size):
                    if self.eval_count >= self.budget:
                        break
                    y = float(func(pop[i]))
                    self.eval_count += 1
                    fitness[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = pop[i].copy()

                stagnation = 0
                continue

            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            # Elitism: preserve top individual
            best_idx = np.argmin(fitness)
            next_pop[0] = pop[best_idx].copy()
            next_fit[0] = fitness[best_idx]
            improved = False

            for i in range(1, self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Tournament selection
                t1 = np.random.choice(self.pop_size, size=2, replace=False)
                p1 = pop[t1[0]] if fitness[t1[0]] < fitness[t1[1]] else pop[t1[1]]

                t2 = np.random.choice(self.pop_size, size=2, replace=False)
                p2 = pop[t2[0]] if fitness[t2[0]] < fitness[t2[1]] else pop[t2[1]]

                # Uniform crossover
                mask = np.random.rand(self.dim) < 0.5
                offspring = np.where(mask, p1, p2)
                offspring = np.clip(offspring, lb, ub)

                y = float(func(offspring))
                self.eval_count += 1

                next_pop[i] = offspring
                next_fit[i] = y

                if y < best_y:
                    best_y = y
                    best_x = offspring.copy()
                    improved = True

            pop = next_pop
            fitness = next_fit

            if improved:
                stagnation = 0
            else:
                stagnation += 1

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
