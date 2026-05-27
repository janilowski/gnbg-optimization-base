# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Firefly Algorithm where candidate solutions attract each other proportional to brightness and inverse square distance.
# Search state: Retains firefly positions, objective light intensities (fitness values), and global best solution.
# Candidate generation: Updates firefly positions by computing attraction vectors towards brighter fireflies combined with Gaussian stochastic noise.
# Selection and replacement: Moves fireflies immediately to new positions if resultant steps remain inside feasible domain bounds.
# Adaptation: Stochastic noise parameter alpha exponentially decays over the search budget to transition from exploration to exploitation.
# Exploration mechanisms: Mutual attraction across all firefly pairs combined with stochastic noise maintains swarm dispersion.
# Exploitation mechanisms: Attraction scaling exponentially favors proximity to the brightest fireflies, concentrating search in elite basins.
# Boundary handling: All firefly attraction steps are explicitly clipped inside valid domain boundaries.
# Budget strategy: Iterates pairwise firefly comparisons while strictly monitoring remaining evaluation budget limits.
# Closest known influences: Firefly Algorithm FA (Yang).
# Novelty or unusual aspects: Pre-computes Euclidean distance scaling gamma to normalize light absorption across arbitrary domain dimensions.
# Failure modes: Quadratic time complexity O(N^2) per iteration limits population size scalability under tight budgets.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 10, max(12, self.dim)))
        if self.pop_size > 30:
            self.pop_size = 30

    def __call__(self, func):
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        self.eval_count = 0
        domain_range = ub - lb
        domain_diag = math.sqrt(np.sum(domain_range ** 2))
        gamma = 1.0 / (domain_diag + 1e-12)

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

        beta0 = 1.0
        alpha_start = 0.2
        alpha_end = 0.01

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            alpha = alpha_start * ((alpha_end / alpha_start) ** progress)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                moved = False

                for j in range(self.pop_size):
                    if self.eval_count >= self.budget:
                        break

                    # Brighter firefly j attracts firefly i (minimization -> smaller fitness is brighter)
                    if fitness[j] < fitness[i]:
                        r = np.linalg.norm(pop[i] - pop[j])
                        beta = beta0 * math.exp(-gamma * (r ** 2))

                        step = np.random.uniform(-0.5, 0.5, size=self.dim) * (alpha * domain_range)
                        trial = pop[i] + beta * (pop[j] - pop[i]) + step
                        trial = np.clip(trial, lb, ub)

                        y = float(func(trial))
                        self.eval_count += 1
                        pop[i] = trial
                        fitness[i] = y
                        moved = True

                        if y < best_y:
                            best_y = y
                            best_x = trial.copy()

                # If i was the brightest, perform random search
                if not moved and self.eval_count < self.budget:
                    step = np.random.uniform(-0.5, 0.5, size=self.dim) * (alpha * domain_range)
                    trial = np.clip(pop[i] + step, lb, ub)
                    y = float(func(trial))
                    self.eval_count += 1
                    pop[i] = trial
                    fitness[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
