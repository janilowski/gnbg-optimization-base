# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Memetic Algorithm combining global Genetic Algorithm recombination with intense local hill climbing around elites.
# Search state: Retains population vectors, objective fitness values, step size parameters, and global best solution.
# Candidate generation: Generates offspring via tournament selection and SBX crossover, followed by local Gaussian hill climbing on the elite offspring.
# Selection and replacement: Offspring generation replaces parent generation with elite preservation; successful local hill climbs replace corresponding offspring.
# Adaptation: Local hill climbing step size scales dynamically based on population diameter.
# Exploration mechanisms: Global tournament selection and SBX recombination maintain search diversity across the domain.
# Exploitation mechanisms: Local Gaussian hill climbing intensifies search precisely around the leading candidate of each generation.
# Boundary handling: All offspring and local hill climbing trial positions are strictly clipped inside domain bounds.
# Budget strategy: Iterates through generational batches and local search steps while strictly checking remaining evaluation budget.
# Closest known influences: Memetic Algorithms / Hybrid Genetic Algorithms (Moscato).
# Novelty or unusual aspects: Restricts local hill climbing specifically to the top generational offspring to maximize budget efficiency.
# Failure modes: Can exhaust evaluation budget rapidly on local climbs if the landscape is highly noisy or deceptive.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 6, max(14, 2 * self.dim)))
        if self.pop_size % 2 != 0:
            self.pop_size += 1
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

        eta_c = 15.0

        while self.eval_count < self.budget:
            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            best_idx = np.argmin(fitness)
            next_pop[0] = pop[best_idx].copy()
            next_fit[0] = fitness[best_idx]

            for i in range(1, self.pop_size, 2):
                if self.eval_count >= self.budget:
                    break

                # Tournament
                t1 = np.random.choice(self.pop_size, size=2, replace=False)
                p1 = pop[t1[0]] if fitness[t1[0]] < fitness[t1[1]] else pop[t1[1]]

                t2 = np.random.choice(self.pop_size, size=2, replace=False)
                p2 = pop[t2[0]] if fitness[t2[0]] < fitness[t2[1]] else pop[t2[1]]

                # SBX
                c1, c2 = p1.copy(), p2.copy()
                if np.random.rand() < 0.9:
                    u = np.random.rand(self.dim)
                    beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta_c + 1.0)), (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)))
                    c1 = np.clip(0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2), lb, ub)
                    c2 = np.clip(0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2), lb, ub)

                y1 = float(func(c1))
                self.eval_count += 1
                next_pop[i] = c1
                next_fit[i] = y1

                if y1 < best_y:
                    best_y = y1
                    best_x = c1.copy()

                if i + 1 < self.pop_size and self.eval_count < self.budget:
                    y2 = float(func(c2))
                    self.eval_count += 1
                    next_pop[i+1] = c2
                    next_fit[i+1] = y2

                    if y2 < best_y:
                        best_y = y2
                        best_x = c2.copy()

            if self.eval_count >= self.budget:
                break

            # Local hill climbing on the best offspring
            best_off_idx = np.argmin(next_fit)
            elite_x = next_pop[best_off_idx].copy()
            elite_y = next_fit[best_off_idx]

            hc_steps = min(5, self.budget - self.eval_count)
            hc_sigma = 0.05 * domain_range

            for _ in range(hc_steps):
                if self.eval_count >= self.budget:
                    break
                step = np.random.normal(0, 1, size=self.dim) * hc_sigma
                trial = np.clip(elite_x + step, lb, ub)
                y = float(func(trial))
                self.eval_count += 1

                if y < elite_y:
                    elite_y = y
                    elite_x = trial.copy()
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()
                else:
                    hc_sigma *= 0.8

            next_pop[best_off_idx] = elite_x
            next_fit[best_off_idx] = elite_y

            pop = next_pop
            fitness = next_fit

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
