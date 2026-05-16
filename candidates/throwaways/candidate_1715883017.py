# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Cuckoo Search algorithm employing Lévy flight random walks to explore continuous parameter optimization spaces.
# Search state: Retains a population of cuckoo nests (candidate solutions), their objective values, and the global best nest.
# Candidate generation: Generates new solutions via heavy-tailed Lévy flight jumps scaled by distance to the global best, or by random mixing of existing nests.
# Selection and replacement: Replaces a randomly chosen nest if the new cuckoo solution improves upon its objective value.
# Adaptation: Discards the worst 25% of nests each generation and replaces them with random exploratory vectors between surviving nests.
# Exploration mechanisms: Heavy-tailed Lévy flights produce occasional extremely large jumps, preventing entrapment in local optima.
# Exploitation mechanisms: Surviving elite nests guide the trajectory scale of Lévy jumps, intensifying search around leading optima.
# Boundary handling: All Lévy flight jumps and exploratory replacements are strictly clipped to stay within domain boundaries.
# Budget strategy: Generates cuckoo flights and nest replacements sequentially, ensuring strict adherence to evaluation budget constraints.
# Closest known influences: Cuckoo Search via Lévy Flights (Yang & Deb).
# Novelty or unusual aspects: Employs exact numerical gamma approximations for stable Mantegna step generation across diverse dimensions.
# Failure modes: Can exhibit slow asymptotic convergence in smooth unimodal basins due to stochastic heavy-tailed perturbations.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 5, max(15, 2 * self.dim)))
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

        pa = 0.25  # Discovery rate of alien eggs
        beta = 1.5
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                   (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Lévy flight step via Mantegna's algorithm
                u = np.random.normal(0, sigma_u, size=self.dim)
                v = np.random.normal(0, 1, size=self.dim)
                step = u / (np.abs(v) ** (1 / beta))

                step_size = 0.01 * step * (pop[i] - best_x)
                trial = pop[i] + step_size * np.random.normal(0, 1, size=self.dim)
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                # Choose random nest j to potentially replace
                j = np.random.randint(self.pop_size)
                if y <= fitness[j]:
                    fitness[j] = y
                    pop[j] = trial
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

            if self.eval_count >= self.budget:
                break

            # Abandon worst nests
            ranks = np.argsort(fitness)
            n_abandon = int(pa * self.pop_size)
            worst_indices = ranks[-n_abandon:]

            for idx in worst_indices:
                if self.eval_count >= self.budget:
                    break
                r1, r2 = np.random.choice(self.pop_size, size=2, replace=False)
                step = np.random.rand(self.dim) * (pop[r1] - pop[r2])
                trial = np.clip(pop[idx] + step, lb, ub)
                y = float(func(trial))
                self.eval_count += 1

                fitness[idx] = y
                pop[idx] = trial
                if y < best_y:
                    best_y = y
                    best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
