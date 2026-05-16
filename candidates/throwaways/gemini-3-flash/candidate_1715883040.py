# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Moth-Flame Optimization (MFO) algorithm modeling navigation around elite flame anchors via logarithmic spirals.
# Search state: Stores moth population positions, objective fitness values, an archive of elite flame positions and fitnesses, and global optimum.
# Candidate generation: Generates updated moth positions by calculating logarithmic spiral trajectories around assigned flame anchors.
# Selection and replacement: Evaluated moths merge with the flame archive; the combined pool is ranked to preserve the top N flames.
# Adaptation: The number of active flames decreases linearly over the search budget, concentrating late search around the single best flame.
# Exploration mechanisms: Individual flame assignments in early iterations disperse moths across diverse local optima.
# Exploitation mechanisms: Logarithmic spiral equations and decreasing active flame counts drive aggressive convergence onto the incumbent optimum.
# Boundary handling: All moth candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates moth positions sequentially per iteration while rigorously verifying evaluation budget limits.
# Closest known influences: Moth-Flame Optimization MFO (Mirjalili).
# Novelty or unusual aspects: Employs dynamic archive merging without external matrix sorting overhead by maintaining a running elite buffer.
# Failure modes: Can experience rapid variance collapse if all initial flames cluster in a single sub-optimal valley.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

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
        best_x = None
        best_y = float("inf")

        moths = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        moth_fit = np.full(self.pop_size, float("inf"))

        flames = np.zeros((self.pop_size, self.dim))
        flame_fit = np.full(self.pop_size, float("inf"))

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = float(func(moths[i]))
            self.eval_count += 1
            moth_fit[i] = y
            if y < best_y:
                best_y = y
                best_x = moths[i].copy()

        # Initialize flames with initial moths
        sorted_idx = np.argsort(moth_fit)
        flames = moths[sorted_idx].copy()
        flame_fit = moth_fit[sorted_idx].copy()

        max_iters = max(1, self.budget // self.pop_size)
        iter_count = 0

        while self.eval_count < self.budget:
            iter_count += 1
            progress = min(1.0, iter_count / max_iters)

            # Number of active flames decreases linearly
            flames_num = int(round(self.pop_size - iter_count * ((self.pop_size - 1) / max_iters)))
            flames_num = max(1, min(flames_num, self.pop_size))

            r_param = -1.0 - progress

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                flame_idx = i if i < flames_num else flames_num - 1
                target_flame = flames[flame_idx]

                distance = np.abs(target_flame - moths[i])
                t = np.random.uniform(r_param, 1.0, size=self.dim)
                b = 1.0

                trial = distance * np.exp(b * t) * np.cos(2.0 * math.pi * t) + target_flame
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1
                moths[i] = trial
                moth_fit[i] = y

                if y < best_y:
                    best_y = y
                    best_x = trial.copy()

            if self.eval_count >= self.budget:
                break

            # Update flames archive: merge moths and flames, take top pop_size
            merged_x = np.vstack((flames, moths))
            merged_fit = np.concatenate((flame_fit, moth_fit))

            sorted_idx = np.argsort(merged_fit)[:self.pop_size]
            flames = merged_x[sorted_idx]
            flame_fit = merged_fit[sorted_idx]

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
