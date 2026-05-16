# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Whale Optimization Algorithm (WOA) modeling encircling prey, bubble-net feeding, and random exploration behaviors.
# Search state: Retains whale population positions, objective fitness values, and global optimum.
# Candidate generation: Updates positions via spiral trajectories towards the global best, linear encircling attraction, or dispersion towards random exemplars.
# Selection and replacement: Evaluated candidate positions replace prior whale positions; global best is updated upon discovering superior solutions.
# Adaptation: Exploration parameter 'a' decreases linearly from 2.0 to 0.0 over the budget to transition from global search to local convergence.
# Exploration mechanisms: Steps towards random whale positions when |A| >= 1 ensure global dispersion across unvisited domain sectors.
# Exploitation mechanisms: Logarithmic spiral bubble-net moves and encircling attraction when |A| < 1 concentrate search in elite basins.
# Boundary handling: All whale candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population positions sequentially in iterations while strictly adhering to evaluation budget ceilings.
# Closest known influences: Whale Optimization Algorithm WOA (Mirjalili & Lewis).
# Novelty or unusual aspects: Combines exact spiral equation scaling with robust vectorized dimension updates.
# Failure modes: Can experience stagnation if the global best anchor is positioned in a sub-optimal deceptive well.
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
            a = 2.0 * (1.0 - progress)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                p = np.random.rand()
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                A = 2.0 * a * r1 - a
                C = 2.0 * r2

                if p < 0.5:
                    if np.linalg.norm(A) < math.sqrt(self.dim): # Average |A| < 1
                        # Encircling prey
                        D = np.abs(C * best_x - pop[i])
                        trial = best_x - A * D
                    else:
                        # Random exploration
                        rand_idx = np.random.randint(self.pop_size)
                        rand_x = pop[rand_idx]
                        D = np.abs(C * rand_x - pop[i])
                        trial = rand_x - A * D
                else:
                    # Spiral bubble-net
                    D_prime = np.abs(best_x - pop[i])
                    l = np.random.uniform(-1, 1)
                    trial = D_prime * math.exp(l) * math.cos(2.0 * math.pi * l) + best_x

                trial = np.clip(trial, lb, ub)
                y = float(func(trial))
                self.eval_count += 1
                pop[i] = trial
                fitness[i] = y

                if y < best_y:
                    best_y = y
                    best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
