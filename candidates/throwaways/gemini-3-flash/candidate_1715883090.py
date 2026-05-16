# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Runge Kutta Optimizer (RKO) employing numerical integration slope estimation formulas to guide search trajectories.
# Search state: Retains agent population positions, objective fitness values, and global optimum across generational iterations.
# Candidate generation: Proposes trial positions by calculating four intermediate Runge-Kutta slopes from random swarm pairs and global best anchors.
# Selection and replacement: Standard one-to-one parent replacement; updates global best optimum upon discovering superior candidate solutions.
# Adaptation: Search step scale parameter alpha decreases exponentially over iterations to refine convergence near discovered basins.
# Exploration mechanisms: Calculating numerical slopes across random population triplets ensures unconstrained multi-directional exploration.
# Exploitation mechanisms: Weighting Runge-Kutta slope terms towards the global best optimum drives steady exploitation in elite valleys.
# Boundary handling: All trial vector positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population agents sequentially in generational iterations while strictly monitoring remaining evaluation budget.
# Closest known influences: Runge Kutta Optimizer RKO (Ahmadianfar et al.).
# Novelty or unusual aspects: Directly embeds standard 4th-order Runge-Kutta ODE numerical integration weighting into mutational updates.
# Failure modes: Slope calculation denominators can experience numerical overflow if position vectors coincide exactly.
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

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            alpha = 0.5 * math.exp(-3.0 * progress)

            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                r1, r2, r3 = np.random.choice(self.pop_size, size=3, replace=False)

                diff12 = pop[r1] - pop[r2]
                diff_norm = np.abs(diff12) + 1e-12

                # Runge-Kutta 4th order intermediate slopes
                k1 = 0.5 * (diff12 / diff_norm) * (best_x - pop[i])
                k2 = 0.5 * ((pop[r1] - pop[r3]) / diff_norm) * (best_x - (pop[i] + k1 * 0.5))
                k3 = 0.5 * ((pop[r2] - pop[r3]) / diff_norm) * (best_x - (pop[i] + k2 * 0.5))
                k4 = 0.5 * ((pop[r1] - pop[i]) / diff_norm) * (best_x - (pop[i] + k3))

                phi = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

                r_chance = np.random.rand()
                if r_chance < 0.5:
                    trial = pop[i] + alpha * phi * domain_range
                else:
                    trial = best_x + alpha * phi * domain_range

                trial = np.clip(trial, lb, ub)
                y = float(func(trial))
                self.eval_count += 1

                if y <= fitness[i]:
                    next_pop[i] = trial
                    next_fit[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()
                else:
                    next_pop[i] = pop[i]
                    next_fit[i] = fitness[i]

            pop = next_pop
            fitness = next_fit

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
