# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Self-Adaptive Differential Evolution algorithm dynamically adjusting strategy probabilities based on empirical historical success rates.
# Search state: Stores active population positions, objective fitness values, strategy probability weights, success/failure counters, and global optimum.
# Candidate generation: Proposes trial vectors by sampling between rand/1/bin and current-to-best/1/bin strategies according to self-adaptive probabilities.
# Selection and replacement: Standard one-to-one parent replacement; records strategy successes and failures over sliding generational windows.
# Adaptation: Periodically recomputes strategy selection probabilities based on exact ratios of successful vs failed trial evaluations.
# Exploration mechanisms: The rand/1/bin strategy maintains wide mutational reach and prevents premature population clustering.
# Exploitation mechanisms: The current-to-best/1/bin strategy actively drives parent trajectories towards the elite incumbent.
# Boundary handling: All trial vector positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population trial vectors sequentially while strictly monitoring remaining evaluation budget limits.
# Closest known influences: Self-Adaptive Differential Evolution SaDE (Qin et al.).
# Novelty or unusual aspects: Pre-computes exact sliding window success/failure ratios to dynamically balance exploration and exploitation.
# Failure modes: Strategy probabilities can experience erratic oscillations if evaluation window thresholds are set too short.
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

        p1 = 0.5
        ns1, ns2 = 0, 0
        nf1, nf2 = 0, 0
        gen = 0
        learn_period = 15

        mu_cr = 0.5
        s_cr = []

        while self.eval_count < self.budget:
            gen += 1
            best_idx = np.argmin(fitness)
            pop_best = pop[best_idx]

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                cr = np.clip(np.random.normal(mu_cr, 0.1), 0.0, 1.0)
                f = np.random.normal(0.5, 0.3)
                while f <= 0:
                    f = np.random.normal(0.5, 0.3)
                f = min(f, 1.0)

                strat_is_1 = (np.random.rand() < p1)
                r1, r2, r3 = np.random.choice(self.pop_size, size=3, replace=False)

                if strat_is_1:
                    # rand/1/bin
                    v = pop[r1] + f * (pop[r2] - pop[r3])
                else:
                    # current-to-best/1/bin
                    v = pop[i] + f * (pop_best - pop[i]) + f * (pop[r1] - pop[r2])

                mask = np.random.rand(self.dim) <= cr
                mask[np.random.randint(self.dim)] = True
                trial = np.clip(np.where(mask, v, pop[i]), lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = y
                    s_cr.append(cr)
                    if strat_is_1:
                        ns1 += 1
                    else:
                        ns2 += 1

                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()
                else:
                    if strat_is_1:
                        nf1 += 1
                    else:
                        nf2 += 1

            if gen % learn_period == 0 and (ns1 + nf1 + ns2 + nf2) > 0:
                rate1 = ns1 / (ns1 + nf1 + 1e-12)
                rate2 = ns2 / (ns2 + nf2 + 1e-12)
                if rate1 + rate2 > 0:
                    p1 = rate1 / (rate1 + rate2)
                p1 = max(0.1, min(0.9, p1))

                ns1, ns2, nf1, nf2 = 0, 0, 0, 0

                if len(s_cr) > 0:
                    mu_cr = np.mean(s_cr)
                    s_cr = []

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
