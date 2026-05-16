# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A JADE-inspired Differential Evolution algorithm utilizing current-to-pbest mutation and an external archive of replaced solutions.
# Search state: Stores active population positions, objective fitness values, external archive of replaced parents, mean F, mean CR, and global optimum.
# Candidate generation: Generates trial vectors via current-to-pbest mutation using random differences drawn from both active population and external archive.
# Selection and replacement: Replaces parent solutions with successful trial vectors and archives the replaced parents into the external storage buffer.
# Adaptation: Dynamic F and CR parameters sampled per individual; mean F and CR adapt via Lehmer mean of successful mutation parameters.
# Exploration mechanisms: Sampling difference vectors from the external archive maintains historical diversity and prevents premature convergence.
# Exploitation mechanisms: Directing mutation steps towards top pbest individuals drives steady convergence into the leading basin.
# Boundary handling: All trial positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population trial vectors sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: JADE with External Archive (Zhang & Sanderson).
# Novelty or unusual aspects: Implements exact Lehmer mean parameter adaptation with dynamic archive pruning in a minimal standalone loop.
# Failure modes: Archive maintenance and pairwise indexing can become computationally burdensome for large populations.
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

        archive = []
        mu_f = 0.5
        mu_cr = 0.5
        c_adapt = 0.1
        p_top = max(2, int(0.1 * self.pop_size))

        while self.eval_count < self.budget:
            s_f = []
            s_cr = []

            # Sort for pbest selection
            sorted_idx = np.argsort(fitness)
            pbest_pool = pop[sorted_idx[:p_top]]

            # Pool for r2 selection (pop + archive)
            if len(archive) > 0:
                union_pool = np.vstack((pop, np.array(archive)))
            else:
                union_pool = pop

            n_union = len(union_pool)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Generate F and CR for individual i
                cr = np.clip(np.random.normal(mu_cr, 0.1), 0.0, 1.0)
                f = np.random.normal(mu_f, 0.1)
                while f <= 0:
                    f = np.random.normal(mu_f, 0.1)
                f = min(f, 1.0)

                # Select pbest
                pbest_idx = np.random.randint(len(pbest_pool))
                pbest = pbest_pool[pbest_idx]

                # Select r1 from pop
                r1 = np.random.randint(self.pop_size)
                while r1 == i:
                    r1 = np.random.randint(self.pop_size)

                # Select r2 from union_pool
                r2 = np.random.randint(n_union)
                while r2 == i or (r2 < self.pop_size and r2 == r1):
                    r2 = np.random.randint(n_union)

                # Mutation: current-to-pbest
                v = pop[i] + f * (pbest - pop[i]) + f * (pop[r1] - union_pool[r2])

                # Binomial crossover
                mask = np.random.rand(self.dim) <= cr
                j_rand = np.random.randint(self.dim)
                mask[j_rand] = True

                trial = np.where(mask, v, pop[i])
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y < fitness[i]:
                    if len(archive) >= self.pop_size:
                        archive.pop(np.random.randint(len(archive)))
                    archive.append(pop[i].copy())

                    pop[i] = trial
                    fitness[i] = y
                    s_f.append(f)
                    s_cr.append(cr)

                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

            if len(s_f) > 0:
                # Lehmer mean for F
                sf_arr = np.array(s_f)
                mu_f = (1.0 - c_adapt) * mu_f + c_adapt * (np.sum(sf_arr ** 2) / (np.sum(sf_arr) + 1e-12))
                
                # Arithmetic mean for CR
                scr_arr = np.array(s_cr)
                mu_cr = (1.0 - c_adapt) * mu_cr + c_adapt * np.mean(scr_arr)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
