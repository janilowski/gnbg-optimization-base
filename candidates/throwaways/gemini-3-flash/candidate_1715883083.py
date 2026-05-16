# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An Equilibrium Optimizer (EO) algorithm inspired by physics mass-balance control volume models to drive optimization search.
# Search state: Stores agent population positions, fitness values, four elite equilibrium pool candidates, their average, and global optimum.
# Candidate generation: Updates agent positions via exponential decay terms and mass generation rates referenced against random equilibrium pool anchors.
# Selection and replacement: Re-sorts population each generation to continuously update the four elite equilibrium pool members.
# Adaptation: Exponential turnover term F contracts search radius as evaluations progress towards total budget exhaustion.
# Exploration mechanisms: Referencing updates against four distinct elite pool members plus their average maintains multi-basin exploratory diversity.
# Exploitation mechanisms: Late iteration decay rates pull agents directly onto the equilibrium pool coordinates in the top optimum basin.
# Boundary handling: All candidate agent positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population agents sequentially in generational iterations while strictly checking remaining evaluation budget.
# Closest known influences: Equilibrium Optimizer EO (Faramarzi et al.).
# Novelty or unusual aspects: Vectorizes exact exponential mass-generation equations across all coordinate dimensions simultaneously.
# Failure modes: Maintaining and sorting four elite pool anchors can cause slight elitist clustering in highly multimodal deceptive landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

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

        a1 = 2.0
        a2 = 1.0
        gp = 0.5

        while self.eval_count < self.budget:
            # Sort for equilibrium pool
            sorted_idx = np.argsort(fitness)
            c_eq1 = pop[sorted_idx[0]]
            c_eq2 = pop[sorted_idx[min(1, self.pop_size - 1)]]
            c_eq3 = pop[sorted_idx[min(2, self.pop_size - 1)]]
            c_eq4 = pop[sorted_idx[min(3, self.pop_size - 1)]]
            c_ave = (c_eq1 + c_eq2 + c_eq3 + c_eq4) / 4.0

            pool = [c_eq1, c_eq2, c_eq3, c_eq4, c_ave]

            progress = self.eval_count / self.budget
            t_scale = (1.0 - progress) ** (a2 * progress)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                c_eq = pool[np.random.randint(len(pool))]

                lam = np.random.uniform(0.001, 1.0, size=self.dim)
                r = np.random.rand(self.dim)
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                f_sign = np.where(r < 0.5, 1.0, -1.0)
                f = a1 * f_sign * (np.exp(-lam * t_scale) - 1.0)

                gcp = np.where(r2 >= gp, 0.5 * r1, 0.0)
                g0 = gcp * (c_eq - pop[i])
                g = g0 * np.exp(-lam * t_scale)

                trial = c_eq + (pop[i] - c_eq) * f + (g / lam) * (1.0 - f)
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
