# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An Arithmetic Optimization Algorithm (AOA) exploring and exploiting domains via fundamental mathematical operators.
# Search state: Retains agent population positions, objective fitness values, and global optimum across generational iterations.
# Candidate generation: Generates trial positions using division and multiplication for global exploration, and addition and subtraction for local exploitation.
# Selection and replacement: Evaluated candidate positions directly replace prior agent coordinates; global best is updated upon discovering superior solutions.
# Adaptation: Math Optimizer Accelerated (MOA) parameter linearly ramps up to smoothly shift swarm from exploratory division to exploitative addition.
# Exploration mechanisms: High Math Optimizer Probability (MOP) and division operators in early iterations generate wide dispersive leaps.
# Exploitation mechanisms: Subtraction and addition operators referenced directly against the global best optimum drive fine local convergence.
# Boundary handling: All agent candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates agent positions sequentially in generational iterations while rigorously checking remaining evaluation budget.
# Closest known influences: Arithmetic Optimization Algorithm AOA (Abualigah et al.).
# Novelty or unusual aspects: Vectorizes exact arithmetic operator choice matrices across all coordinate dimensions simultaneously.
# Failure modes: Division operators can cause extreme coordinate jumps if denominator regularization epsilon is set too small.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

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

        moa_min, moa_max = 0.2, 0.9
        alpha, mu_param = 5.0, 0.05

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            moa = moa_min + progress * (moa_max - moa_min)
            mop = 1.0 - (progress) ** (1.0 / alpha)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                r1 = np.random.rand()
                r2 = np.random.rand(self.dim)
                r3 = np.random.rand(self.dim)
                mu_vec = np.random.uniform(0.0, 1.0, size=self.dim)

                step_scale = domain_range * mu_param * mu_vec

                if r1 > moa:
                    # Exploration: Division or Multiplication
                    div_step = best_x / (mop + 1e-6) * step_scale
                    mul_step = best_x * mop * step_scale
                    trial = np.where(r2 < 0.5, pop[i] + div_step, pop[i] + mul_step)
                else:
                    # Exploitation: Subtraction or Addition
                    sub_step = best_x - mop * step_scale
                    add_step = best_x + mop * step_scale
                    trial = np.where(r3 < 0.5, sub_step, add_step)

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
