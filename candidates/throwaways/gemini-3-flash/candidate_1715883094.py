# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Comprehensive Learning Particle Swarm Optimizer (CLPSO) incorporating Cauchy mutations to dislodge stagnating particles.
# Search state: Retains agent positions, velocities, personal best positions and fitnesses, stagnation counters, exemplar tables, and global optimum.
# Candidate generation: Updates velocities using coordinate-wise exemplar learning from peer personal bests, with Cauchy mutations upon stagnation.
# Selection and replacement: Standard personal best updates; updates global best optimum whenever a candidate achieves superior objective fitness.
# Adaptation: Learning probabilities are pre-assigned across the swarm; exemplar tables are regenerated upon particle stagnation.
# Exploration mechanisms: Learning coordinate exemplars from distinct peer personal bests maintains high diversity across decoupled dimensions.
# Exploitation mechanisms: Exemplars selected via tournament comparisons ensure velocity trajectories favor superior personal best attractors.
# Boundary handling: All agent candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates swarm members sequentially in generational iterations while strictly checking remaining evaluation budget.
# Closest known influences: Comprehensive Learning PSO CLPSO (Liang et al.).
# Novelty or unusual aspects: Directly embeds Cauchy stochastic resets upon individual particle stagnation to bypass local trapping wells.
# Failure modes: Maintaining coordinate exemplars and stagnation counters per particle increases memory and indexing overhead.
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
        v_max = 0.2 * domain_range

        best_x = None
        best_y = float("inf")

        pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        vel = np.random.uniform(-v_max, v_max, size=(self.pop_size, self.dim))
        pbest_x = pop.copy()
        pbest_y = np.full(self.pop_size, float("inf"))
        stagnation = np.zeros(self.pop_size, dtype=int)

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop[i]))
            self.eval_count += 1
            pbest_y[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Learning probabilities Pc
        ranks = np.arange(self.pop_size)
        pc = 0.05 + 0.45 * (np.exp(10.0 * ranks / (max(1, self.pop_size - 1))) - 1.0) / (math.exp(10.0) - 1.0)

        def generate_exemplars(agent_idx):
            ex = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < pc[agent_idx]:
                    t1, t2 = np.random.choice(self.pop_size, size=2, replace=False)
                    best_t = t1 if pbest_y[t1] < pbest_y[t2] else t2
                    ex[j] = pbest_x[best_t, j]
                else:
                    ex[j] = pbest_x[agent_idx, j]
            return ex

        exemplars = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            exemplars[i] = generate_exemplars(i)

        w_start, w_end = 0.9, 0.4
        c_param = 1.5

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            w = w_start - (w_start - w_end) * progress

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                if stagnation[i] > 7:
                    # Cauchy mutation reset
                    u = np.random.uniform(0.001, 0.999, size=self.dim)
                    cauchy_step = np.tan(np.pi * (u - 0.5))
                    pop[i] = np.clip(pbest_x[i] + 0.1 * domain_range * cauchy_step, lb, ub)
                    stagnation[i] = 0
                    exemplars[i] = generate_exemplars(i)
                else:
                    r1 = np.random.rand(self.dim)
                    vel[i] = w * vel[i] + c_param * r1 * (exemplars[i] - pop[i])
                    vel[i] = np.clip(vel[i], -v_max, v_max)
                    pop[i] += vel[i]

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
                    if stagnation[i] > 5:
                        exemplars[i] = generate_exemplars(i)

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
