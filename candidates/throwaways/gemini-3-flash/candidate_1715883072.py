# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A SHADE-inspired Differential Evolution algorithm utilizing success-history parameter adaptation memory tables.
# Search state: Stores active population positions, objective fitness values, historical memory buffers for F and CR, and global optimum.
# Candidate generation: Generates trial vectors via current-to-pbest mutation using Cauchy-sampled F and Gaussian-sampled CR parameters.
# Selection and replacement: Replaces parent solutions with successful trial vectors; tracks objective improvement magnitudes for parameter updates.
# Adaptation: Updates historical memory tables via weighted Lehmer mean of F and weighted arithmetic mean of CR based on improvement magnitudes.
# Exploration mechanisms: Heavy-tailed Cauchy sampling for F parameters enables occasional long exploratory leaps across the domain.
# Exploitation mechanisms: Directing mutation steps towards top pbest individuals drives steady convergence into elite basins.
# Boundary handling: All trial positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population trial vectors sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Success-History Based Parameter Adaptation for Differential Evolution SHADE (Tanabe & Fukunaga).
# Novelty or unusual aspects: Combines weighted historical memory tables with robust inverse-transform Cauchy parameter generation.
# Failure modes: Can experience parameter stagnation if all memory tables collapse into identical unreactive values during early iterations.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

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

        H = 6
        M_cr = np.full(H, 0.5)
        M_f = np.full(H, 0.5)
        k_mem = 0
        p_top = max(2, int(0.1 * self.pop_size))

        while self.eval_count < self.budget:
            s_f = []
            s_cr = []
            s_delta = []

            sorted_idx = np.argsort(fitness)
            pbest_pool = pop[sorted_idx[:p_top]]

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                r_idx = np.random.randint(H)
                mu_cr = M_cr[r_idx]
                mu_f = M_f[r_idx]

                # Sample CR from Gaussian
                cr = np.clip(np.random.normal(mu_cr, 0.1), 0.0, 1.0)

                # Sample F from Cauchy: tan(pi * (u - 0.5))
                u = np.random.uniform(0.001, 0.999)
                cauchy_step = math.tan(math.pi * (u - 0.5))
                f = mu_f + 0.1 * cauchy_step
                while f <= 0:
                    u = np.random.uniform(0.001, 0.999)
                    f = mu_f + 0.1 * math.tan(math.pi * (u - 0.5))
                f = min(f, 1.0)

                pbest_idx = np.random.randint(len(pbest_pool))
                pbest = pbest_pool[pbest_idx]

                r1 = np.random.randint(self.pop_size)
                while r1 == i:
                    r1 = np.random.randint(self.pop_size)

                r2 = np.random.randint(self.pop_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(self.pop_size)

                v = pop[i] + f * (pbest - pop[i]) + f * (pop[r1] - pop[r2])

                mask = np.random.rand(self.dim) <= cr
                j_rand = np.random.randint(self.dim)
                mask[j_rand] = True

                trial = np.where(mask, v, pop[i])
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y < fitness[i]:
                    s_f.append(f)
                    s_cr.append(cr)
                    s_delta.append(abs(fitness[i] - y))

                    pop[i] = trial
                    fitness[i] = y

                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

            if len(s_f) > 0:
                weights = np.array(s_delta) / (np.sum(s_delta) + 1e-12)
                sf_arr = np.array(s_f)
                scr_arr = np.array(s_cr)

                # Weighted Lehmer mean for F
                M_f[k_mem] = np.sum(weights * (sf_arr ** 2)) / (np.sum(weights * sf_arr) + 1e-12)
                
                # Weighted arithmetic mean for CR
                M_cr[k_mem] = np.sum(weights * scr_arr)

                k_mem = (k_mem + 1) % H

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
