# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Grey Wolf Optimizer (GWO) augmenting standard social hierarchy hunting equations with an historical evolutionary memory buffer.
# Search state: Retains agent population positions, fitness values, alpha/beta/delta leader positions, historical memory buffer, and global optimum.
# Candidate generation: Proposes positions via weighted attraction towards alpha/beta/delta pack leaders and sampled historical elite exemplars.
# Selection and replacement: Re-sorts pack each generation to update leadership hierarchy; archives alpha leaders into historical memory.
# Adaptation: Encircling parameter 'a' linearly decreases from 2 to 0 over iterations to smoothly transition pack from exploration to exploitation.
# Exploration mechanisms: Referencing hunting vectors against archived historical exemplars prevents pack insularity and explores multiple basins.
# Exploitation mechanisms: Attraction vectors towards current alpha and beta leaders aggressively pull pack into the leading optimum valley.
# Boundary handling: All agent candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population agents sequentially in generational iterations while strictly monitoring remaining evaluation budget.
# Closest known influences: Grey Wolf Optimizer GWO (Mirjalili et al.) / Memory-based GWO.
# Novelty or unusual aspects: Directly replaces standard delta wolf attraction terms with historical memory exemplars during exploratory phases.
# Failure modes: Pack leadership hierarchy can cluster tightly in single basins if memory buffer sampling rates are set too low.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

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

        alpha_pos, beta_pos, delta_pos = None, None, None
        memory = []
        max_memory = 30

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            a_param = 2.0 * (1.0 - progress)

            sorted_idx = np.argsort(fitness)
            alpha_pos = pop[sorted_idx[0]].copy()
            beta_pos = pop[sorted_idx[min(1, self.pop_size - 1)]].copy()
            delta_pos = pop[sorted_idx[min(2, self.pop_size - 1)]].copy()

            if len(memory) < max_memory:
                memory.append(alpha_pos.copy())
            else:
                memory[np.random.randint(max_memory)] = alpha_pos.copy()

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Target 1: Alpha
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A1 = 2.0 * a_param * r1 - a_param
                C1 = 2.0 * r2
                D_alpha = np.abs(C1 * alpha_pos - pop[i])
                X1 = alpha_pos - A1 * D_alpha

                # Target 2: Beta
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A2 = 2.0 * a_param * r1 - a_param
                C2 = 2.0 * r2
                D_beta = np.abs(C2 * beta_pos - pop[i])
                X2 = beta_pos - A2 * D_beta

                # Target 3: Memory or Delta
                mem_target = memory[np.random.randint(len(memory))] if np.random.rand() < 0.5 else delta_pos
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                A3 = 2.0 * a_param * r1 - a_param
                C3 = 2.0 * r2
                D_delta = np.abs(C3 * mem_target - pop[i])
                X3 = mem_target - A3 * D_delta

                trial = (X1 + X2 + X3) / 3.0
                trial = np.clip(trial, lb, ub)

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
