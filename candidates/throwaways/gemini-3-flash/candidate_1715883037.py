# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Gravitational Search Algorithm (GSA) where candidate solutions act as masses attracting each other via Newtonian physics.
# Search state: Retains agent positions, velocities, inertial masses computed from objective rankings, and global optimum.
# Candidate generation: Updates velocities and positions using cumulative gravitational acceleration vectors from elite attracting masses.
# Selection and replacement: Moves agents to resultant positions each iteration; recalculates gravitational masses based on updated objective rankings.
# Adaptation: Gravitational constant G decays exponentially over the search budget to shift from global attraction to local exploitation.
# Exploration mechanisms: Mutual attraction across disparate mass positions in early iterations maintains global exploration.
# Exploitation mechanisms: Heavy elite masses dominate the acceleration field in late iterations, drawing the swarm into the leading optimum.
# Boundary handling: All agent positions are clipped inside domain bounds, and velocities are inverted upon boundary collisions.
# Budget strategy: Evaluates agent positions sequentially per iteration while rigorously verifying evaluation budget limits.
# Closest known influences: Gravitational Search Algorithm GSA (Rashedi et al.).
# Novelty or unusual aspects: Restricts gravitational attraction specifically to K-best masses to prevent numerical cancellation in dense swarms.
# Failure modes: Can experience numerical instability if distance metrics between collapsing elites become excessively small.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 10, max(12, self.dim)))
        if self.pop_size > 30:
            self.pop_size = 30

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
        vel = np.zeros((self.pop_size, self.dim))
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

        G0 = 100.0
        alpha = 20.0
        max_iters = max(1, self.budget // self.pop_size)
        iter_count = 0

        while self.eval_count < self.budget:
            iter_count += 1
            G = G0 * np.exp(-alpha * (iter_count / max_iters))

            # Calculate masses
            worst = np.max(fitness)
            best = np.min(fitness)
            if abs(best - worst) < 1e-12:
                mass = np.ones(self.pop_size)
            else:
                mass = (worst - fitness) / (worst - best + 1e-12)
            
            total_mass = np.sum(mass) + 1e-12
            M = mass / total_mass

            # Focus attraction on K-best agents
            kbest = max(2, int(self.pop_size * (1.0 - iter_count / max_iters)))
            sorted_idx = np.argsort(fitness)
            elite_indices = sorted_idx[:kbest]

            acc = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                for j in elite_indices:
                    if i == j:
                        continue
                    dist = np.linalg.norm(pop[i] - pop[j])
                    force = G * (M[j]) / (dist + 1e-8)  # Acceleration directly
                    r = np.random.rand(self.dim)
                    acc[i] += r * force * (pop[j] - pop[i])

                vel[i] = np.random.rand(self.dim) * vel[i] + acc[i]
                pop[i] += vel[i]

                # Boundary handling
                out = (pop[i] < lb) | (pop[i] > ub)
                vel[i][out] = -0.5 * vel[i][out]
                pop[i] = np.clip(pop[i], lb, ub)

                y = float(func(pop[i]))
                self.eval_count += 1
                fitness[i] = y

                if y < best_y:
                    best_y = y
                    best_x = pop[i].copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
