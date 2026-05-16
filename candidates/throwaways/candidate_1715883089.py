# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Colony Predation Algorithm (CPA) segregating the swarm into communicating predators chasing optimal basins and stochastically dispersing prey.
# Search state: Retains agent population positions, objective fitness values, predator/prey partition indices, and global optimum.
# Candidate generation: Elite predators update via attraction towards the global best and peer communication; inferior prey disperse stochastically across the domain.
# Selection and replacement: Standard one-to-one parent replacement; re-sorts swarm each generation to assign predator and prey roles.
# Adaptation: Dispersal step magnitude exponentially decays over iterations to transition prey from global jumpers to local searchers.
# Exploration mechanisms: Stochastic dispersion of bottom-tier prey maintains constant mutational reach across unvisited landscape valleys.
# Exploitation mechanisms: Predator attraction vectors towards the elite global best ensure rapid convergence within discovered optimal basins.
# Boundary handling: All predator and prey candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population agents sequentially in generational iterations while strictly monitoring remaining evaluation budget.
# Closest known influences: Colony Predation Algorithm / Predator-Prey Optimization.
# Novelty or unusual aspects: Directly embeds role switching based on exact median fitness partitioning each generation.
# Failure modes: Can experience sluggish convergence if prey dispersion vectors persistently jump out of valid boundary constraints.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 8, max(20, 2 * self.dim)))
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

        half_pop = self.pop_size // 2

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            decay = math.exp(-5.0 * progress)

            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]

            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                if i < half_pop:
                    # Predator regime
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    pr1, pr2 = np.random.choice(half_pop, size=2, replace=False)

                    step = r1 * (best_x - pop[i]) + r2 * (pop[pr1] - pop[pr2])
                    trial = pop[i] + step
                else:
                    # Prey regime
                    z = np.random.normal(0, 1, size=self.dim)
                    pred_idx = np.random.randint(half_pop)
                    step = decay * z * domain_range * 0.1 + 0.1 * np.random.rand(self.dim) * (pop[i] - pop[pred_idx])
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
