# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An adaptive Differential Evolution algorithm based on JADE utilizing current-to-pbest mutation and parameter tracking.
# Search state: Retains population vectors, objective values, global best, and historical adaptive means for F and CR parameters.
# Candidate generation: Generates trial vectors by shifting current positions towards a top 15% pbest elite member and adding a difference vector.
# Selection and replacement: Parent vectors are replaced by trial vectors if objective values improve or remain equal.
# Adaptation: Successful F and CR parameters are collected each generation to update historical distribution means (Cauchy for F, Normal for CR).
# Exploration mechanisms: Cauchy distribution sampling for F occasionally produces large mutation steps to escape local optima.
# Exploitation mechanisms: Direct attraction vector towards pbest elite members concentrates search around leading basins.
# Boundary handling: All trial positions are bounded by clipping to the search space domain.
# Budget strategy: Iterates through generational population loops while verifying budget exhaustion before every evaluation.
# Closest known influences: JADE (Zhang & Sanderson).
# Novelty or unusual aspects: Simplified external archive tracking to maintain minimal memory footprint and fast runtime execution.
# Failure modes: Can experience parameter drift in highly deceptive functions if successful steps are fortuitous outliers.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 6, max(12, 3 * self.dim)))
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

        mu_cr = 0.5
        mu_f = 0.5
        c = 0.1
        p_ratio = max(1, int(0.15 * self.pop_size))

        while self.eval_count < self.budget:
            s_cr = []
            s_f = []

            # Sort population for pbest selection
            sorted_idx = np.argsort(fitness)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Generate CRi and Fi
                cr_i = np.random.normal(mu_cr, 0.1)
                cr_i = np.clip(cr_i, 0.0, 1.0)

                # Cauchy distribution for Fi
                f_i = np.random.standard_cauchy() * 0.1 + mu_f
                while f_i <= 0:
                    f_i = np.random.standard_cauchy() * 0.1 + mu_f
                if f_i > 1:
                    f_i = 1.0

                # Select pbest
                pbest_idx = np.random.choice(sorted_idx[:p_ratio])
                
                # Select r1, r2 distinct
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(candidates, size=2, replace=False)

                mutant = pop[i] + f_i * (pop[pbest_idx] - pop[i]) + f_i * (pop[r1] - pop[r2])

                # Binomial Crossover
                cross_mask = np.random.rand(self.dim) < cr_i
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(self.dim)] = True

                trial = np.where(cross_mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y <= fitness[i]:
                    s_cr.append(cr_i)
                    s_f.append(f_i)
                    fitness[i] = y
                    pop[i] = trial
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

            if s_cr:
                mu_cr = (1 - c) * mu_cr + c * np.mean(s_cr)
            if s_f:
                mu_f = (1 - c) * mu_f + c * (np.sum(np.array(s_f)**2) / (np.sum(s_f) + 1e-12))

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
