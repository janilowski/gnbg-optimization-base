# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Bat Algorithm modeling echolocation behavior with pulse rate and loudness adaptation.
# Search state: Retains bat positions, velocities, frequencies, pulse emission rates, loudness values, and global optimum.
# Candidate generation: Updates velocities using variable frequencies towards the global best, with occasional local Gaussian flights around the global best.
# Selection and replacement: Replaces parent solutions if trial flights achieve equal or superior fitness and satisfy loudness acceptance criteria.
# Adaptation: Employs exponential loudness decay and pulse emission rate expansion as iterations progress.
# Exploration mechanisms: Frequency variation and initial high loudness maintain wide exploratory trajectories.
# Exploitation mechanisms: Stochastic local walks around the global optimum drive fine-tuning convergence in the leading basin.
# Boundary handling: All bat positions and local search steps are strictly clipped inside valid domain boundaries.
# Budget strategy: Generates bat flights sequentially per iteration while rigorously verifying evaluation budget limits.
# Closest known influences: Bat Algorithm BA (Yang).
# Novelty or unusual aspects: Combines exact frequency modulation with continuous Gaussian local walks scaled by mean loudness.
# Failure modes: Can experience parameter drift if loudness decays too rapidly before discovering the global basin.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 8, max(15, 2 * self.dim)))
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
        domain_range = ub - lb

        best_x = None
        best_y = float("inf")

        pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        vel = np.zeros((self.pop_size, self.dim))
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

        f_min = 0.0
        f_max = 2.0
        A = np.full(self.pop_size, 1.0)
        r0 = 0.5
        r = np.full(self.pop_size, r0)
        alpha = 0.9
        gamma = 0.1
        iter_count = 0

        while self.eval_count < self.budget:
            iter_count += 1
            mean_A = np.mean(A)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                freq = f_min + (f_max - f_min) * np.random.rand()
                vel[i] = vel[i] + (pop[i] - best_x) * freq
                trial = pop[i] + vel[i]

                # Local search around best_x
                if np.random.rand() > r[i]:
                    step = np.random.normal(0, 1, size=self.dim) * (0.1 * mean_A * domain_range)
                    trial = best_x + step

                trial = np.clip(trial, lb, ub)
                y = float(func(trial))
                self.eval_count += 1

                # Acceptance check
                if y <= fitness[i] and np.random.rand() < A[i]:
                    fitness[i] = y
                    pop[i] = trial
                    A[i] *= alpha
                    r[i] = r0 * (1.0 - np.exp(-gamma * iter_count))

                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
