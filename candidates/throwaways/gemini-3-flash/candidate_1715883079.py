# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Modified Differential Evolution algorithm replacing deterministic difference vectors with coordinate Gaussian sampling scaled by population differences.
# Search state: Retains active population positions, objective fitness values, and global optimum across generational iterations.
# Candidate generation: Generates trial vectors by sampling Gaussian distributions centered on pbest anchors with standard deviations proportional to coordinate differences.
# Selection and replacement: Standard one-to-one parent replacement; ranks population each generation to establish elite pbest pools.
# Adaptation: Coordinate standard deviations automatically scale down as population difference vectors contract around optimal basins.
# Exploration mechanisms: Gaussian stochastic sampling around difference vectors prevents grid alignment artifacts and maintains multi-directional exploration.
# Exploitation mechanisms: Centering Gaussian mutation distributions directly on top pbest individuals drives aggressive local basin convergence.
# Boundary handling: All trial vector positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population trial vectors sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Modified Differential Evolution with p-best crossover / MDE_pBX (Islam et al.).
# Novelty or unusual aspects: Directly embeds exact coordinate-wise Gaussian variance scaling into standard DE mutational equations.
# Failure modes: Can exhibit slower convergence on strictly linear diagonal slopes compared to deterministic vector additions.
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

        p_top = max(2, int(0.15 * self.pop_size))
        f, cr = 0.8, 0.8
        min_std = 1e-6 * domain_range

        while self.eval_count < self.budget:
            sorted_idx = np.argsort(fitness)
            pbest_pool = pop[sorted_idx[:p_top]]

            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                pbest = pbest_pool[np.random.randint(len(pbest_pool))]

                r1 = np.random.randint(self.pop_size)
                while r1 == i:
                    r1 = np.random.randint(self.pop_size)

                r2 = np.random.randint(self.pop_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(self.pop_size)

                # Gaussian mutation scaled by |pop[r1] - pop[r2]|
                std = np.maximum(f * np.abs(pop[r1] - pop[r2]), min_std)
                z = np.random.normal(0, 1, size=self.dim)
                v = pbest + std * z

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
