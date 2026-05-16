# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Slime Mould Algorithm (SMA) modeling bio-oscillation venous network formation to balance global exploration and local food source exploitation.
# Search state: Retains agent population positions, objective fitness values, bio-oscillator weight tables, and global optimum.
# Candidate generation: Proposes positions via weighted difference vectors between swarm members and bio-oscillatory contractions towards the elite best.
# Selection and replacement: Evaluated candidate positions directly replace prior agent coordinates; global best is updated upon discovering superior solutions.
# Adaptation: Bio-oscillation weights adapt dynamically based on log-normalized fitness rankings across the swarm.
# Exploration mechanisms: Occasional unconstrained random leaps across domain bounds prevent stagnation in local attraction basins.
# Exploitation mechanisms: Bio-oscillation weights heavily amplify attraction vectors towards top elite agents, pulling swarm into optimal food sources.
# Boundary handling: All agent candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates agent positions sequentially in generational iterations while rigorously checking remaining evaluation budget.
# Closest known influences: Slime Mould Algorithm SMA (Li et al.).
# Novelty or unusual aspects: Vectorizes exact log-normalized bio-oscillator weight computations across the sorted swarm.
# Failure modes: Parameter scaling contractions (vc) can cause over-aggressive origin clustering if unreferenced against the optimum.
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

        z_prob = 0.03
        half_pop = self.pop_size // 2

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            sorted_idx = np.argsort(fitness)
            pop = pop[sorted_idx]
            fitness = fitness[sorted_idx]

            worst_y = fitness[-1]
            best_curr_y = fitness[0]

            # Compute bio-oscillator weights W
            weights = np.ones((self.pop_size, self.dim))
            denom = best_curr_y - worst_y + 1e-12

            for i in range(self.pop_size):
                condition = (fitness[i] - best_curr_y) / denom
                r_val = np.random.rand(self.dim)
                if i < half_pop:
                    weights[i] = 1.0 + r_val * math.log(abs(condition) + 1.0)
                else:
                    weights[i] = 1.0 - r_val * math.log(abs(condition) + 1.0)

            a_param = math.atanh(max(0.001, min(0.999, 1.0 - progress)))

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                vb = np.random.uniform(-a_param, a_param, size=self.dim)
                vc = np.random.uniform(-1.0, 1.0, size=self.dim) * (1.0 - progress)

                if np.random.rand() < z_prob:
                    trial = np.random.uniform(lb, ub, size=self.dim)
                else:
                    p_chance = math.tanh(abs(fitness[i] - best_y))
                    r_chance = np.random.rand(self.dim)

                    r1, r2 = np.random.choice(self.pop_size, size=2, replace=False)
                    step_w = best_x + vb * (weights[i] * pop[r1] - pop[r2])
                    step_c = best_x + vc * (pop[i] - best_x)

                    trial = np.where(r_chance < p_chance, step_w, step_c)

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
