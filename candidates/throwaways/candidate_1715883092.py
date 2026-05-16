# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Dynamic Multi-Swarm Particle Swarm algorithm dividing agents into communicating sub-swarms with periodic random regrouping.
# Search state: Retains agent positions, velocities, personal bests, sub-swarm assignments, sub-swarm bests, and global optimum.
# Candidate generation: Agents update velocities and positions referenced against personal bests and their assigned sub-swarm best anchor.
# Selection and replacement: Standard personal best updates; periodic random regrouping completely re-allocates particles into new sub-swarms.
# Adaptation: Inertia weight decreases linearly to damp velocity oscillations as iterations progress towards total budget exhaustion.
# Exploration mechanisms: Random regrouping every R generations destroys local sub-swarm insularity and transfers information across distant basins.
# Exploitation mechanisms: Sub-swarm best velocity attractions allow small particle groups to rapidly fine-tune local optimal valleys.
# Boundary handling: All agent positions are explicitly clipped inside domain bounds; velocities are inverted upon boundary collisions.
# Budget strategy: Evaluates multi-swarm members sequentially in generational iterations while checking remaining evaluation budget.
# Closest known influences: Dynamic Multi-Swarm Particle Swarm Optimizer DMS-PSO (Liang & Suganthan).
# Novelty or unusual aspects: Directly embeds sub-swarm indexing and periodic permutation shuffling inside a unified vectorized loop.
# Failure modes: Can experience sluggish global convergence if sub-swarm size is too small to maintain stable velocity trajectories.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.n_swarms = max(3, min(6, self.dim // 2))
        self.sub_size = max(5, min(10, self.budget // (10 * self.n_swarms)))
        self.pop_size = self.n_swarms * self.sub_size

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

        # Initial sub-swarm assignment
        indices = np.arange(self.pop_size)
        np.random.shuffle(indices)
        swarms = indices.reshape((self.n_swarms, self.sub_size))

        w_start, w_end = 0.9, 0.4
        c1, c2 = 1.5, 1.5
        gen = 0
        regroup_period = 15

        while self.eval_count < self.budget:
            gen += 1
            progress = self.eval_count / self.budget
            w = w_start - (w_start - w_end) * progress

            # Regroup check
            if gen % regroup_period == 0:
                np.random.shuffle(indices)
                swarms = indices.reshape((self.n_swarms, self.sub_size))

            # Find sub-swarm bests
            lbest_x = np.zeros((self.n_swarms, self.dim))
            for k in range(self.n_swarms):
                sub_idx = swarms[k]
                best_sub_idx = sub_idx[np.argmin(pbest_y[sub_idx])]
                lbest_x[k] = pbest_x[best_sub_idx].copy()

            for k in range(self.n_swarms):
                for idx in swarms[k]:
                    if self.eval_count >= self.budget:
                        break

                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)

                    vel[idx] = w * vel[idx] + c1 * r1 * (pbest_x[idx] - pop[idx]) + c2 * r2 * (lbest_x[k] - pop[idx])
                    vel[idx] = np.clip(vel[idx], -v_max, v_max)
                    pop[idx] += vel[idx]

                    out = (pop[idx] < lb) | (pop[idx] > ub)
                    vel[idx][out] = -0.5 * vel[idx][out]
                    pop[idx] = np.clip(pop[idx], lb, ub)

                    y = float(func(pop[idx]))
                    self.eval_count += 1

                    if y <= pbest_y[idx]:
                        pbest_y[idx] = y
                        pbest_x[idx] = pop[idx].copy()
                        if y < best_y:
                            best_y = y
                            best_x = pop[idx].copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
