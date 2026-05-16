# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An Adaptive Covariance Hill Climbing algorithm screening local Gaussian mutation pools and adapting Cholesky matrices along successful trajectories.
# Search state: Retains incumbent solution position, objective fitness value, Cholesky factor matrix, global step size, and global optimum.
# Candidate generation: Generates candidate mutation pools via matrix multiplication of the Cholesky factor and standard normal vectors.
# Selection and replacement: Replaces incumbent position whenever the best candidate in the local pool achieves equal or superior objective fitness.
# Adaptation: Step size expands or contracts based on pool success rates; Cholesky factor matrix updates directly along successful jump directions.
# Exploration mechanisms: Unconstrained Cholesky matrix updates allow local search corridors to align smoothly along arbitrary rotated ridges.
# Exploitation mechanisms: Pool selection and multiplicative step size contractions rapidly pinpoint the exact floor of discovered local valleys.
# Boundary handling: All candidate pool positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates candidate mutation pools sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Covariance Matrix Adaptation Hill Climbing / (mu+1)-CMA-ES.
# Novelty or unusual aspects: Directly embeds exact rank-one Cholesky multiplicative updates into a local pool screening hill climber.
# Failure modes: Can stall or make sluggish progress if local pool size is too small to find successful descent vectors on narrow ridges.
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

        sigma = 0.25
        min_sigma = 1e-6
        max_sigma = 1.0
        A = np.diag(domain_range.copy())
        c_cov = min(0.5, 2.0 / (self.dim ** 2 + 6.0))

        stagnation = 0

        while self.eval_count < self.budget:
            n_pool = min(8, max(4, self.dim // 2))
            pool_x = np.zeros((n_pool, self.dim))
            pool_z = np.zeros((n_pool, self.dim))
            fitness = np.full(n_pool, float("inf"))

            for i in range(n_pool):
                if self.eval_count >= self.budget:
                    break

                z = np.random.normal(0, 1, size=self.dim)
                step = A @ z
                cand = np.clip(curr_x + sigma * step, lb, ub)

                y = float(func(cand))
                self.eval_count += 1

                pool_x[i] = cand
                pool_z[i] = z
                fitness[i] = y

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            if self.eval_count >= self.budget:
                break

            best_idx = np.argmin(fitness)
            min_y = fitness[best_idx]

            if min_y < curr_y:
                curr_x = pool_x[best_idx].copy()
                curr_y = min_y

                z0 = pool_z[best_idx]
                norm_z0_sq = np.dot(z0, z0)
                if norm_z0_sq > 1e-10:
                    factor = math.sqrt(1.0 - c_cov)
                    term = (math.sqrt(1.0 + (c_cov / (1.0 - c_cov)) * norm_z0_sq) - 1.0) / norm_z0_sq
                    Az = A @ z0
                    A = factor * A + (factor * term) * np.outer(Az, z0)

                sigma = min(sigma * 1.2, max_sigma)
                stagnation = 0
            else:
                sigma = max(sigma * 0.8, min_sigma)
                stagnation += 1

            if stagnation > 25 or sigma < min_sigma * 5 or np.max(np.abs(A)) > 1e10:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                sigma = 0.25
                A = np.diag(domain_range.copy())
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
