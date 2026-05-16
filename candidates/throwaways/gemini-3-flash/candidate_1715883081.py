# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Hybrid Swarm algorithm partitioning agents into concurrent Particle Swarm and Differential Evolution optimization regimes.
# Search state: Retains agent positions, velocities, personal best positions and fitnesses, and global optimum across iterations.
# Candidate generation: Half the swarm updates via velocity attraction towards personal and global bests; half mutates via DE difference vectors.
# Selection and replacement: Evaluated agents update personal best positions whenever trial coordinates achieve equal or superior fitness.
# Adaptation: Inertia weight linearly contracts over the search budget to damp velocity trajectories as iterations progress.
# Exploration mechanisms: The DE regime maintains difference vector diversity across personal bests, preventing premature swarm collapse.
# Exploitation mechanisms: The PSO regime aggressively accelerates particles towards the global best anchor in the leading basin.
# Boundary handling: All agent positions are explicitly clipped inside domain bounds; velocities are inverted upon boundary collisions.
# Budget strategy: Evaluates swarm members sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Hybrid PSO-DE (Liu et al.).
# Novelty or unusual aspects: Directly shares personal best position archives across distinct velocity update and difference mutation operators.
# Failure modes: Can experience parameter imbalance if the DE crossover rate is poorly matched to the landscape separability.
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

        w_start, w_end = 0.9, 0.4
        c1, c2 = 1.5, 1.5
        f, cr = 0.8, 0.8
        half_pop = self.pop_size // 2

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            w = w_start - (w_start - w_end) * progress

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                if i < half_pop:
                    # PSO regime
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    vel[i] = w * vel[i] + c1 * r1 * (pbest_x[i] - pop[i]) + c2 * r2 * (best_x - pop[i])
                    vel[i] = np.clip(vel[i], -v_max, v_max)
                    pop[i] += vel[i]

                    # Boundary collision bounce
                    out = (pop[i] < lb) | (pop[i] > ub)
                    vel[i][out] = -0.5 * vel[i][out]
                    pop[i] = np.clip(pop[i], lb, ub)
                else:
                    # DE regime using pbest_x as base pool
                    r1, r2, r3 = np.random.choice(self.pop_size, size=3, replace=False)
                    v = pbest_x[r1] + f * (pbest_x[r2] - pbest_x[r3])

                    mask = np.random.rand(self.dim) <= cr
                    mask[np.random.randint(self.dim)] = True

                    pop[i] = np.clip(np.where(mask, v, pop[i]), lb, ub)

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
