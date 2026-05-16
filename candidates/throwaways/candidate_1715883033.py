# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Quantum-Behaved Particle Swarm Optimization (QPSO) algorithm replacing velocity vectors with quantum delta potential well sampling.
# Search state: Retains particle positions, personal best positions and fitness values, and global optimum.
# Candidate generation: Updates positions by sampling exponential distributions centered on stochastic local attractors and scaled by distance to the swarm mean best position.
# Selection and replacement: Updates personal best positions whenever candidate positions yield superior objective values.
# Adaptation: Contraction-expansion parameter alpha decreases linearly from 1.0 to 0.5 to transition from global dispersion to local convergence.
# Exploration mechanisms: Exponential sampling allows infinite theoretical reach across all coordinates, preventing absolute entrapment.
# Exploitation mechanisms: Attraction towards the global best and personal best positions centers quantum wells around elite basins.
# Boundary handling: All generated candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates swarm members sequentially in iterations while strictly adhering to evaluation budget ceilings.
# Closest known influences: Quantum-Behaved Particle Swarm Optimization QPSO (Sun et al.).
# Novelty or unusual aspects: Completely eliminates velocity vector tracking, saving memory and avoiding hyperparameter velocity clamping.
# Failure modes: Can experience premature collapse if alpha decays too rapidly on highly multimodal deceptive landscapes.
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

        pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        pbest_x = pop.copy()
        pbest_y = np.full(self.pop_size, float("inf"))

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop[i]))
            self.eval_count += 1
            pbest_y[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        alpha_start = 1.0
        alpha_end = 0.5

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            alpha = alpha_start - (alpha_start - alpha_end) * progress

            # Mean best position
            mbest = np.mean(pbest_x, axis=0)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                phi = np.random.rand(self.dim)
                p_attractor = phi * pbest_x[i] + (1.0 - phi) * best_x

                u = np.random.rand(self.dim)
                u = np.clip(u, 1e-12, 1.0)
                L = alpha * np.abs(mbest - pop[i])

                direction = np.random.choice([-1, 1], size=self.dim)
                trial = p_attractor + direction * L * np.log(1.0 / u)
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1
                pop[i] = trial

                if y < pbest_y[i]:
                    pbest_y[i] = y
                    pbest_x[i] = trial.copy()
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
