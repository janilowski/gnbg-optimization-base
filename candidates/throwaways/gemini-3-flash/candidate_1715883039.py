# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Salp Swarm Algorithm (SSA) structuring population agents as a chain following an elite food source leader.
# Search state: Retains salp chain positions, objective fitness values, and food source (global best) coordinates.
# Candidate generation: Leader updates position via stochastic oscillations around the food source; followers average positions with preceding chain members.
# Selection and replacement: Evaluated candidate positions replace prior chain positions; population is re-sorted each iteration to update leader/follower roles.
# Adaptation: Exploration coefficient c1 exponentially contracts over the search budget to damp leader oscillations around the food source.
# Exploration mechanisms: Stochastic bounding step expansions in the leader position maintain global reach across the domain.
# Exploitation mechanisms: Follower averaging creates a smooth hydrodynamic convergence corridor drawing the entire chain towards the leader.
# Boundary handling: All salp candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates leader and follower positions sequentially while strictly checking remaining evaluation budget.
# Closest known influences: Salp Swarm Algorithm SSA (Mirjalili et al.).
# Novelty or unusual aspects: Employs continuous chain averaging to eliminate independent random walks among follower agents.
# Failure modes: Can experience sluggish chain adaptation if the leader repeatedly jumps across opposite sides of a deceptive ridge.
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

        # Sort salps initially
        sorted_idx = np.argsort(fitness)
        pop = pop[sorted_idx]
        fitness = fitness[sorted_idx]

        max_iters = max(1, self.budget // self.pop_size)
        iter_count = 0

        while self.eval_count < self.budget:
            iter_count += 1
            c1 = 2.0 * math.exp(-((4.0 * iter_count / max_iters) ** 2))

            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            # --- Leader Salp Update (i = 0) ---
            if self.eval_count < self.budget:
                c2 = np.random.rand(self.dim)
                c3 = np.random.rand(self.dim)

                step = c1 * (domain_range * c2 + lb)
                leader_cand = np.where(c3 >= 0.5, best_x + step, best_x - step)
                leader_cand = np.clip(leader_cand, lb, ub)

                y = float(func(leader_cand))
                self.eval_count += 1
                next_pop[0] = leader_cand
                next_fit[0] = y

                if y < best_y:
                    best_y = y
                    best_x = leader_cand.copy()

            # --- Follower Salps Update (i >= 1) ---
            for i in range(1, self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Average with preceding salp
                follower_cand = 0.5 * (pop[i] + pop[i - 1])
                follower_cand = np.clip(follower_cand, lb, ub)

                y = float(func(follower_cand))
                self.eval_count += 1
                next_pop[i] = follower_cand
                next_fit[i] = y

                if y < best_y:
                    best_y = y
                    best_x = follower_cand.copy()

            # Re-sort salp chain
            sorted_idx = np.argsort(next_fit)
            pop = next_pop[sorted_idx]
            fitness = next_fit[sorted_idx]

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
