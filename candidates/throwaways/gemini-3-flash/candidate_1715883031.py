# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Comprehensive Learning Particle Swarm Optimization (CLPSO) algorithm where particles learn from personal bests of diverse swarm exemplars per dimension.
# Search state: Retains particle positions, velocities, personal best positions and fitnesses, learning probabilities, and global best.
# Candidate generation: Updates velocities using exemplar target vectors assembled coordinate-by-coordinate from personal bests of tournament-selected particles.
# Selection and replacement: Particles update their personal best positions whenever new positions achieve superior fitness.
# Adaptation: Inertia weight linearly decreases from 0.9 to 0.4 over the search budget.
# Exploration mechanisms: Particle-specific learning probabilities (Pc) cause particles to learn from different exemplars on different dimensions, avoiding premature swarm collapse.
# Exploitation mechanisms: Exemplar tournaments favor highly fit personal best positions, driving steady swarm convergence.
# Boundary handling: Positions are clipped inside domain bounds, and velocities are inverted and dampened upon boundary collisions.
# Budget strategy: Evaluates swarm members sequentially while strictly checking remaining evaluation budget limits.
# Closest known influences: Comprehensive Learning PSO (Liang et al.).
# Novelty or unusual aspects: Pre-computes exponential learning probability distribution across particles to establish stable exploratory diversity.
# Failure modes: Can exhibit slower initial convergence on simple unimodal functions compared to standard global-best PSO.
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
        domain_range = ub - lb
        v_max = 0.2 * domain_range

        best_x = None
        best_y = float("inf")

        pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        vel = np.random.uniform(-v_max, v_max, size=(self.pop_size, self.dim))
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

        # Compute learning probabilities Pc
        indices = np.arange(self.pop_size)
        pc = 0.05 + 0.45 * (np.exp(10.0 * indices / (max(1, self.pop_size - 1))) - 1.0) / (math.exp(10.0) - 1.0)

        w_start, w_end = 0.9, 0.4
        c = 1.49445

        # Refresh gap for exemplars
        refresh_gap = 7
        stagnation = np.zeros(self.pop_size, dtype=int)
        exemplars = np.zeros((self.pop_size, self.dim), dtype=int)

        # Initialize exemplars
        for i in range(self.pop_size):
            for j in range(self.dim):
                if np.random.rand() < pc[i]:
                    t1, t2 = np.random.choice(self.pop_size, size=2, replace=False)
                    exemplars[i, j] = t1 if pbest_y[t1] < pbest_y[t2] else t2
                else:
                    exemplars[i, j] = i

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            w = w_start - (w_start - w_end) * progress

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Check if exemplar update is needed
                if stagnation[i] > refresh_gap:
                    for j in range(self.dim):
                        if np.random.rand() < pc[i]:
                            t1, t2 = np.random.choice(self.pop_size, size=2, replace=False)
                            exemplars[i, j] = t1 if pbest_y[t1] < pbest_y[t2] else t2
                        else:
                            exemplars[i, j] = i
                    stagnation[i] = 0

                # Assemble target vector
                target_pbest = np.array([pbest_x[exemplars[i, j], j] for j in range(self.dim)])

                r = np.random.rand(self.dim)
                vel[i] = w * vel[i] + c * r * (target_pbest - pop[i])
                vel[i] = np.clip(vel[i], -v_max, v_max)
                pop[i] += vel[i]

                # Boundary handling
                out = (pop[i] < lb) | (pop[i] > ub)
                vel[i][out] = -0.5 * vel[i][out]
                pop[i] = np.clip(pop[i], lb, ub)

                y = float(func(pop[i]))
                self.eval_count += 1

                if y < pbest_y[i]:
                    pbest_y[i] = y
                    pbest_x[i] = pop[i].copy()
                    stagnation[i] = 0
                    if y < best_y:
                        best_y = y
                        best_x = pop[i].copy()
                else:
                    stagnation[i] += 1

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
