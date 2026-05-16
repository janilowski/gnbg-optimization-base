# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust Particle Swarm Optimization (PSO) implementation featuring linearly decreasing inertia weight and velocity clamping.
# Search state: Maintains swarm positions, velocities, personal best positions and fitnesses, and the global best solution.
# Candidate generation: Particles move through the search space by updating their velocities towards personal best and global best locations.
# Selection and replacement: Each particle evaluates its new position; if fitness improves, personal best is updated. Global best tracks the overall swarm optimum.
# Adaptation: Inertia weight decreases linearly over the evaluation iterations from 0.9 to 0.4 to transition from exploration to exploitation.
# Exploration mechanisms: Initial randomized velocities and cognitive exploration towards personal bests maintain swarm diversity.
# Exploitation mechanisms: Social attraction towards the global best accelerates convergence in promising areas.
# Boundary handling: Positions are clamped to the valid box domain, and velocities are zeroed out when boundary collisions occur.
# Budget strategy: Generational loops evaluate swarm members sequentially until the total evaluation budget is exhausted.
# Closest known influences: Standard Particle Swarm Optimization with inertia weight (Shi & Eberhart).
# Novelty or unusual aspects: Dynamic velocity clamping scaled to the search space domain preventing explosive particle trajectories.
# Failure modes: Susceptible to premature convergence in highly multimodal deceptive wells.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 10, max(15, 2 * self.dim)))
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

        domain_range = ub - lb
        v_max = 0.2 * domain_range

        # Initialize positions and velocities
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

        c1 = 1.8  # Cognitive parameter
        c2 = 1.8  # Social parameter

        w_start = 0.9
        w_end = 0.4

        while self.eval_count < self.budget:
            # Calculate current inertia weight
            progress = self.eval_count / self.budget
            w = w_start - (w_start - w_end) * progress

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                vel[i] = w * vel[i] + c1 * r1 * (pbest_x[i] - pop[i]) + c2 * r2 * (best_x - pop[i])
                vel[i] = np.clip(vel[i], -v_max, v_max)

                pop[i] = pop[i] + vel[i]

                # Boundary handling
                out_of_bounds = (pop[i] < lb) | (pop[i] > ub)
                vel[i][out_of_bounds] = -0.5 * vel[i][out_of_bounds] # Dampen velocity
                pop[i] = np.clip(pop[i], lb, ub)

                y = float(func(pop[i]))
                self.eval_count += 1

                if y < pbest_y[i]:
                    pbest_y[i] = y
                    pbest_x[i] = pop[i].copy()
                    if y < best_y:
                        best_y = y
                        best_x = pop[i].copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
