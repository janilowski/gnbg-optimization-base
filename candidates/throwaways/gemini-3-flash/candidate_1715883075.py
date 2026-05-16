# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Composite Differential Evolution algorithm combining three distinct mutation strategies and parameter pairs per parent.
# Search state: Retains active population positions, objective fitness values, and global optimum across generations.
# Candidate generation: Generates three competing trial vectors per parent using rand/1, rand/2, and current-to-best strategies paired with diverse F/CR values.
# Selection and replacement: Evaluates all three candidate trial vectors and replaces the parent solution if the superior trial achieves improvement.
# Adaptation: Simultaneously applies aggressive exploratory and exploitative operators to adaptively match local landscape curvature.
# Exploration mechanisms: The rand/2 strategy combined with F=1.0 maintains strong mutational reach and prevents population entrapment.
# Exploitation mechanisms: The current-to-best strategy paired with high CR drives aggressive convergence towards the leading optimum.
# Boundary handling: All generated trial vectors are explicitly clipped inside valid domain boundaries.
# Budget strategy: Allocates evaluations in triplets per parent while strictly monitoring remaining evaluation budget.
# Closest known influences: Composite Differential Evolution CoDE (Wang et al.).
# Novelty or unusual aspects: Simultaneously evaluates three candidate vectors per parent without requiring dynamic parameter tracking or memory tables.
# Failure modes: Tripling evaluations per parent reduces the number of generations achievable under tight evaluation budgets.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 12, max(15, self.dim)))
        if self.pop_size < 6:
            self.pop_size = 6
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

        param_pairs = [(1.0, 0.1), (1.0, 0.9), (0.8, 0.2)]

        while self.eval_count < self.budget:
            best_idx = np.argmin(fitness)
            pop_best = pop[best_idx]

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Random permutation of param pairs for the 3 strategies
                perm_params = [param_pairs[k] for k in np.random.permutation(3)]

                r1, r2, r3, r4, r5 = np.random.choice(self.pop_size, size=5, replace=False)

                trials = []

                # Strategy 1: rand/1/bin
                f1, cr1 = perm_params[0]
                v1 = pop[r1] + f1 * (pop[r2] - pop[r3])
                mask1 = np.random.rand(self.dim) <= cr1
                mask1[np.random.randint(self.dim)] = True
                u1 = np.clip(np.where(mask1, v1, pop[i]), lb, ub)
                trials.append(u1)

                # Strategy 2: rand/2/bin
                f2, cr2 = perm_params[1]
                v2 = pop[r1] + f2 * (pop[r2] - pop[r3]) + f2 * (pop[r4] - pop[r5])
                mask2 = np.random.rand(self.dim) <= cr2
                mask2[np.random.randint(self.dim)] = True
                u2 = np.clip(np.where(mask2, v2, pop[i]), lb, ub)
                trials.append(u2)

                # Strategy 3: current-to-best/1/bin
                f3, cr3 = perm_params[2]
                v3 = pop[i] + f3 * (pop_best - pop[i]) + f3 * (pop[r1] - pop[r2])
                mask3 = np.random.rand(self.dim) <= cr3
                mask3[np.random.randint(self.dim)] = True
                u3 = np.clip(np.where(mask3, v3, pop[i]), lb, ub)
                trials.append(u3)

                best_trial_x = None
                best_trial_y = float("inf")

                for u in trials:
                    if self.eval_count >= self.budget:
                        break
                    uy = float(func(u))
                    self.eval_count += 1

                    if uy < best_trial_y:
                        best_trial_y = uy
                        best_trial_x = u.copy()

                    if uy < best_y:
                        best_y = uy
                        best_x = u.copy()

                if best_trial_y < fitness[i] and best_trial_x is not None:
                    pop[i] = best_trial_x
                    fitness[i] = best_trial_y

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
