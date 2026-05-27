# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Multi-Population Differential Evolution algorithm running concurrent exploratory and exploitative sub-populations with periodic migration.
# Search state: Retains positions and objective values for two distinct sub-populations, along with migration counters and global optimum.
# Candidate generation: Sub-population 1 generates trial vectors via rand/1/bin (exploration); Sub-population 2 generates trial vectors via best/1/bin (exploitation).
# Selection and replacement: Standard one-to-one parent replacement per sub-population; periodic migration exchanges best and worst individuals across sub-populations.
# Adaptation: Concurrently maintains global search diversity and rapid local basin convergence through distinct sub-population parameterizations.
# Exploration mechanisms: Sub-population 1 (F=0.9, CR=0.1) performs wide mutational steps across coordinates to discover new basins.
# Exploitation mechanisms: Sub-population 2 (F=0.5, CR=0.9) actively clusters around the elite best solution to refine local optima.
# Boundary handling: All trial vector positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates sub-population trial vectors sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Multi-Population Differential Evolution / Island Model Optimization.
# Novelty or unusual aspects: Directly embeds dual-island migration dynamics into a minimal single-process generation loop.
# Failure modes: Can split evaluation budgets inefficiently if one sub-population gets trapped in a deceptive basin early.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.sub_pop = int(min(self.budget // 16, max(15, self.dim)))
        if self.sub_pop > 35:
            self.sub_pop = 35

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

        pop1 = np.random.uniform(lb, ub, size=(self.sub_pop, self.dim))
        fit1 = np.full(self.sub_pop, float("inf"))

        pop2 = np.random.uniform(lb, ub, size=(self.sub_pop, self.dim))
        fit2 = np.full(self.sub_pop, float("inf"))

        for i in range(self.sub_pop):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop1[i]))
            self.eval_count += 1
            fit1[i] = y
            if y < best_y:
                best_y = y
                best_x = pop1[i].copy()

            if self.eval_count >= self.budget:
                break
            y2 = float(func(pop2[i]))
            self.eval_count += 1
            fit2[i] = y2
            if y2 < best_y:
                best_y = y2
                best_x = pop2[i].copy()

        gen = 0
        mig_interval = 10

        while self.eval_count < self.budget:
            gen += 1

            # --- Sub-population 1: DE/rand/1/bin (Exploration: F=0.9, CR=0.1) ---
            f1, cr1 = 0.9, 0.1
            for i in range(self.sub_pop):
                if self.eval_count >= self.budget:
                    break
                r1, r2, r3 = np.random.choice(self.sub_pop, size=3, replace=False)
                v = pop1[r1] + f1 * (pop1[r2] - pop1[r3])
                mask = np.random.rand(self.dim) <= cr1
                mask[np.random.randint(self.dim)] = True
                trial = np.clip(np.where(mask, v, pop1[i]), lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y < fit1[i]:
                    fit1[i] = y
                    pop1[i] = trial
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

            # --- Sub-population 2: DE/best/1/bin (Exploitation: F=0.5, CR=0.9) ---
            if self.eval_count >= self.budget:
                break
            best2_idx = np.argmin(fit2)
            pop2_best = pop2[best2_idx]
            f2, cr2 = 0.5, 0.9

            for i in range(self.sub_pop):
                if self.eval_count >= self.budget:
                    break
                r1, r2 = np.random.choice(self.sub_pop, size=2, replace=False)
                v = pop2_best + f2 * (pop2[r1] - pop2[r2])
                mask = np.random.rand(self.dim) <= cr2
                mask[np.random.randint(self.dim)] = True
                trial = np.clip(np.where(mask, v, pop2[i]), lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y < fit2[i]:
                    fit2[i] = y
                    pop2[i] = trial
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

            # --- Migration ---
            if gen % mig_interval == 0 and self.eval_count < self.budget:
                b1_idx = np.argmin(fit1)
                w1_idx = np.argmax(fit1)
                b2_idx = np.argmin(fit2)
                w2_idx = np.argmax(fit2)

                # Swap elite copies into worst slots
                pop1[w1_idx] = pop2[b2_idx].copy()
                fit1[w1_idx] = fit2[b2_idx]

                pop2[w2_idx] = pop1[b1_idx].copy()
                fit2[w2_idx] = fit1[b1_idx]

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
