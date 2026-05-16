# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Sine Cosine Algorithm (SCA) proposing candidate positions via trigonometric oscillations towards the global best solution.
# Search state: Retains agent population positions, objective fitness values, and global optimum across generational iterations.
# Candidate generation: Proposes trial vectors by scaling distance vectors towards the global best using oscillating sine and cosine functions.
# Selection and replacement: Evaluated candidate positions directly replace prior agent coordinates; global best is updated upon discovering superior solutions.
# Adaptation: Oscillating amplitude parameter r1 decreases linearly from 2.0 to 0.0 to smoothly transition from exploration to exploitation.
# Exploration mechanisms: High amplitude r1 and stochastic multipliers r3 in early iterations push agents outwards across unvisited domain sectors.
# Exploitation mechanisms: Low amplitude r1 in late iterations contracts trigonometric oscillations tightly around the global best anchor.
# Boundary handling: All agent candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates agent positions sequentially in generational iterations while rigorously checking remaining evaluation budget.
# Closest known influences: Sine Cosine Algorithm SCA (Mirjalili).
# Novelty or unusual aspects: Employs vectorized trigonometric updates across coordinates to eliminate nested looping overhead.
# Failure modes: Can experience premature convergence if the global best anchor is positioned in a sub-optimal deceptive well.
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

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop[i]))
            self.eval_count += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        a_param = 2.0

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            r1 = a_param * (1.0 - progress)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                r2 = np.random.uniform(0.0, 2.0 * math.pi, size=self.dim)
                r3 = np.random.uniform(0.0, 2.0, size=self.dim)
                r4 = np.random.rand(self.dim)

                diff = np.abs(r3 * best_x - pop[i])
                step_sin = r1 * np.sin(r2) * diff
                step_cos = r1 * np.cos(r2) * diff

                trial = pop[i] + np.where(r4 < 0.5, step_sin, step_cos)
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1
                pop[i] = trial

                if y < best_y:
                    best_y = y
                    best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
