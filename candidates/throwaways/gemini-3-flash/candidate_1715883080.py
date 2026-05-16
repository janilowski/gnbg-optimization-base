# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Dual-Mutation Differential Evolution algorithm evaluating concurrent exploratory and exploitative trial vectors for each parent solution.
# Search state: Retains active population positions, objective fitness values, and global optimum across generational iterations.
# Candidate generation: Proposes two competing trial vectors per parent using rand/1/bin (exploration) and current-to-best/1/bin (exploitation).
# Selection and replacement: Evaluates both competing trial vectors and replaces the parent solution if the superior trial achieves improvement.
# Adaptation: Concurrently maintains wide global difference exploration and focused convergence towards the leading optimum.
# Exploration mechanisms: The rand/1/bin trial vector ensures unconstrained mutational reach across unvisited landscape sectors.
# Exploitation mechanisms: The current-to-best/1/bin trial vector actively pulls parent trajectories towards the elite incumbent.
# Boundary handling: All trial vector positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Allocates evaluations in pairs per parent while strictly monitoring remaining evaluation budget.
# Closest known influences: Composite Differential Evolution / Multi-Strategy DE.
# Novelty or unusual aspects: Directly pits exploratory and exploitative operators against each other in pairwise duels per parent.
# Failure modes: Doubling evaluations per parent reduces total generational iterations under fixed evaluation ceilings.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 10, max(15, self.dim)))
        if self.pop_size < 8:
            self.pop_size = 8
        if self.pop_size > 40:
            self.pop_size = 40

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
            best_idx = np.argmin(fitness)
            pop_best = pop[best_idx]

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                r1, r2, r3 = np.random.choice(self.pop_size, size=3, replace=False)

                # Trial 1: rand/1/bin
                v1 = pop[r1] + f * (pop[r2] - pop[r3])
                mask1 = np.random.rand(self.dim) <= cr
                mask1[np.random.randint(self.dim)] = True
                u1 = np.clip(np.where(mask1, v1, pop[i]), lb, ub)

                y1 = float(func(u1))
                self.eval_count += 1

                if y1 < best_y:
                    best_y = y1
                    best_x = u1.copy()

                if self.eval_count >= self.budget:
                    if y1 < fitness[i]:
                        pop[i] = u1
                        fitness[i] = y1
                    break

                # Trial 2: current-to-best/1/bin
                r4, r5 = np.random.choice(self.pop_size, size=2, replace=False)
                v2 = pop[i] + f * (pop_best - pop[i]) + f * (pop[r4] - pop[r5])
                mask2 = np.random.rand(self.dim) <= cr
                mask2[np.random.randint(self.dim)] = True
                u2 = np.clip(np.where(mask2, v2, pop[i]), lb, ub)

                y2 = float(func(u2))
                self.eval_count += 1

                if y2 < best_y:
                    best_y = y2
                    best_x = u2.copy()

                if y1 <= y2 and y1 < fitness[i]:
                    pop[i] = u1
                    fitness[i] = y1
                elif y2 < y1 and y2 < fitness[i]:
                    pop[i] = u2
                    fitness[i] = y2

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
