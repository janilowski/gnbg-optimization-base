# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Stochastic Hill Climbing algorithm utilizing heavy-tailed Cauchy mutations for local and semi-global exploration.
# Search state: Retains incumbent solution position, objective fitness value, coordinate step size vector, and global optimum.
# Candidate generation: Generates neighborhood candidate batches via Cauchy distributed random jumps scaled by current step sizes.
# Selection and replacement: Moves to the superior neighborhood candidate if its objective fitness improves upon the incumbent.
# Adaptation: Multiplicatively expands step size upon successful moves and contracts step size upon local search failures.
# Exploration mechanisms: Heavy tails of the Cauchy distribution produce occasional long leaps, enabling escape from local minima basins.
# Exploitation mechanisms: Small step contractions during stagnation fine-tune the incumbent solution around the local minimum.
# Boundary handling: All Cauchy jump candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates neighborhood candidate batches sequentially while strictly checking remaining evaluation budget.
# Closest known influences: Fast Evolutionary Programming / Cauchy Hill Climbing (Yao et al.).
# Novelty or unusual aspects: Combines exact inverse transform Cauchy sampling with dynamic success-based step size scaling.
# Failure modes: Can exhibit slow asymptotic convergence in smooth convex basins compared to Gaussian or gradient methods.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0

    def __call__(self, func):
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        self.eval_count = 0
        domain_range = ub - lb

        curr_x = np.random.uniform(lb, ub, size=self.dim)
        curr_y = float(func(curr_x))
        self.eval_count += 1

        best_x = curr_x.copy()
        best_y = curr_y

        sigma = 0.1 * domain_range
        min_sigma = 1e-6 * domain_range
        max_sigma = 0.4 * domain_range
        stagnation = 0
        neigh_size = min(10, max(4, self.dim))

        while self.eval_count < self.budget:
            best_neigh_x = None
            best_neigh_y = float("inf")

            for _ in range(neigh_size):
                if self.eval_count >= self.budget:
                    break

                # Inverse transform sampling for Cauchy distribution: tan(pi * (u - 0.5))
                u = np.random.uniform(0.001, 0.999, size=self.dim)
                cauchy_step = np.tan(math.pi * (u - 0.5))
                
                trial = np.clip(curr_x + cauchy_step * sigma, lb, ub)
                y = float(func(trial))
                self.eval_count += 1

                if y < best_neigh_y:
                    best_neigh_y = y
                    best_neigh_x = trial.copy()

                if y < best_y:
                    best_y = y
                    best_x = trial.copy()

            if self.eval_count >= self.budget:
                break

            if best_neigh_y < curr_y:
                curr_x = best_neigh_x.copy()
                curr_y = best_neigh_y
                sigma = np.minimum(sigma * 1.1, max_sigma)
                stagnation = 0
            else:
                sigma = np.maximum(sigma * 0.85, min_sigma)
                stagnation += 1

            if stagnation > 25 or np.max(sigma / domain_range) < 1e-5:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                sigma = 0.1 * domain_range
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
