# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An L-SHADE optimization algorithm integrating success-history parameter adaptation with linear population size reduction.
# Search state: Retains active population positions, objective fitness values, historical memory buffers for F and CR, external archive, and global optimum.
# Candidate generation: Generates trial vectors via current-to-pbest mutation using Cauchy-sampled F and Gaussian-sampled CR parameters.
# Selection and replacement: Replaces parent solutions with successful trial vectors; prunes worst population members each generation to match linear decay schedule.
# Adaptation: Historical memory tables adapt via weighted means of successful parameters; population size decreases linearly from N_init down to N_min.
# Exploration mechanisms: Large initial population size and difference vectors drawn from external archive maintain robust global exploration.
# Exploitation mechanisms: Linear population size reduction shifts evaluation budget entirely towards elite individuals during late iterations.
# Boundary handling: All trial positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population trial vectors sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: L-SHADE (Tanabe & Fukunaga).
# Novelty or unusual aspects: Directly embeds dynamic population truncation and memory table updates into a unified compact loop.
# Failure modes: Can truncate population too rapidly if evaluation budgets are exceptionally small relative to dimensionality.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_init = int(min(self.budget // 8, max(25, 2 * self.dim)))
        if self.pop_init > 60:
            self.pop_init = 60
        self.pop_min = max(4, self.dim // 2)

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

        pop_size = self.pop_init
        pop = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        fitness = np.full(pop_size, float("inf"))

        for i in range(pop_size):
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
        archive = []

        while self.eval_count < self.budget:
            s_f = []
            s_cr = []
            s_delta = []

            p_top = max(2, int(0.1 * pop_size))
            sorted_idx = np.argsort(fitness)
            pbest_pool = pop[sorted_idx[:p_top]]

            if len(archive) > 0:
                union_pool = np.vstack((pop, np.array(archive)))
            else:
                union_pool = pop
            n_union = len(union_pool)

            for i in range(pop_size):
                if self.eval_count >= self.budget:
                    break

                r_idx = np.random.randint(H)
                mu_cr = M_cr[r_idx]
                mu_f = M_f[r_idx]

                cr = np.clip(np.random.normal(mu_cr, 0.1), 0.0, 1.0)
                u = np.random.uniform(0.001, 0.999)
                cauchy_step = math.tan(math.pi * (u - 0.5))
                f = mu_f + 0.1 * cauchy_step
                while f <= 0:
                    u = np.random.uniform(0.001, 0.999)
                    f = mu_f + 0.1 * math.tan(math.pi * (u - 0.5))
                f = min(f, 1.0)

                pbest_idx = np.random.randint(len(pbest_pool))
                pbest = pbest_pool[pbest_idx]

                r1 = np.random.randint(pop_size)
                while r1 == i:
                    r1 = np.random.randint(pop_size)

                r2 = np.random.randint(n_union)
                while r2 == i or (r2 < pop_size and r2 == r1):
                    r2 = np.random.randint(n_union)

                v = pop[i] + f * (pbest - pop[i]) + f * (pop[r1] - union_pool[r2])

                mask = np.random.rand(self.dim) <= cr
                j_rand = np.random.randint(self.dim)
                mask[j_rand] = True

                trial = np.where(mask, v, pop[i])
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y < fitness[i]:
                    if len(archive) >= pop_size:
                        archive.pop(np.random.randint(len(archive)))
                    archive.append(pop[i].copy())

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

                M_f[k_mem] = np.sum(weights * (sf_arr ** 2)) / (np.sum(weights * sf_arr) + 1e-12)
                M_cr[k_mem] = np.sum(weights * scr_arr)
                k_mem = (k_mem + 1) % H

            if self.eval_count >= self.budget:
                break

            # Linear population size reduction
            progress = self.eval_count / self.budget
            target_pop = int(round((self.pop_min - self.pop_init) * progress + self.pop_init))
            target_pop = max(self.pop_min, min(target_pop, pop_size))

            if target_pop < pop_size:
                sorted_idx = np.argsort(fitness)[:target_pop]
                pop = pop[sorted_idx]
                fitness = fitness[sorted_idx]
                pop_size = target_pop
                if len(archive) > pop_size:
                    archive = archive[:pop_size]

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
