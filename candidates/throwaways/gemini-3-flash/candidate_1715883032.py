# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Fully Informed Particle Swarm (FIPS) algorithm where particles update trajectories using personal bests of all topological neighbors.
# Search state: Stores particle positions, velocities, personal best positions and fitness values, and global optimum.
# Candidate generation: Updates velocities by computing a stochastic center of gravity across personal best positions of immediate ring topology neighbors.
# Selection and replacement: Particles update personal best positions whenever new positions yield superior or equal objective values.
# Adaptation: Employs constriction coefficient scaling to ensure asymptotic trajectory stability without strict velocity clamping.
# Exploration mechanisms: Localized ring topology interactions maintain swarm diversity across multiple niches in the landscape.
# Exploitation mechanisms: Aggregating attraction vectors from multiple successful neighbors drives robust convergence towards local basins.
# Boundary handling: Positions are clipped inside domain bounds, and velocities are inverted upon boundary collisions.
# Budget strategy: Evaluates swarm members sequentially in iterations while strictly adhering to evaluation budget ceilings.
# Closest known influences: Fully Informed Particle Swarm FIPS (Mendes et al.).
# Novelty or unusual aspects: Combines exact constriction coefficient weighting with continuous ring neighborhood consensus.
# Failure modes: Can stall or oscillate in complex curved valleys if neighbor attraction vectors repeatedly cancel out.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

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
        v_max = 0.5 * domain_range

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

        # Constriction coefficient parameters
        phi = 4.1
        chi = 0.72984
        neigh_size = 3  # self, left, right

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Neighbors in ring topology
                neighbors = [(i - 1) % self.pop_size, i, (i + 1) % self.pop_size]

                # FIPS consensus target
                u_sum = np.zeros(self.dim)
                p_fips = np.zeros(self.dim)

                for n_idx in neighbors:
                    u_k = np.random.uniform(0, phi / neigh_size, size=self.dim)
                    u_sum += u_k
                    p_fips += u_k * pbest_x[n_idx]

                p_fips = np.where(u_sum > 0, p_fips / u_sum, pop[i])

                # Velocity update
                vel[i] = chi * (vel[i] + u_sum * (p_fips - pop[i]))
                vel[i] = np.clip(vel[i], -v_max, v_max)
                pop[i] += vel[i]

                # Boundary handling
                out = (pop[i] < lb) | (pop[i] > ub)
                vel[i][out] = -0.5 * vel[i][out]
                pop[i] = np.clip(pop[i], lb, ub)

                y = float(func(pop[i]))
                self.eval_count += 1

                if y <= pbest_y[i]:
                    pbest_y[i] = y
                    pbest_x[i] = pop[i].copy()
                    if y < best_y:
                        best_y = y
                        best_x = pop[i].copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
