# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous (1+1)-CMA-ES algorithm adapting covariance Cholesky factors directly from successful elitist mutational jumps.
# Search state: Retains incumbent solution position, objective fitness value, Cholesky factor matrix, global step size, and success probability.
# Candidate generation: Proposes a single offspring per iteration via matrix multiplication of the Cholesky factor and a standard normal vector.
# Selection and replacement: Replaces the incumbent parent solution if the offspring achieves equal or superior objective fitness.
# Adaptation: Adapts global step size via the 1/5th success rule and updates the Cholesky factor along successful jump directions.
# Exploration mechanisms: Unconstrained Cholesky matrix updates allow exploration trajectories to align along arbitrary rotated valleys.
# Exploitation mechanisms: Strict (1+1) elitism ensures monotonic objective improvement throughout the search budget.
# Boundary handling: All offspring candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates single offspring sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: (1+1)-CMA-ES (Igel et al.).
# Novelty or unusual aspects: Integrates Cholesky factor updates directly into a minimal one-parent, one-offspring elitist loop.
# Failure modes: Can get trapped in local minima or false ridges if single stochastic jumps cannot cross deceptive barriers.
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

        sigma = 0.2
        A = np.diag(domain_range.copy())
        min_sigma = 1e-6

        p_succ = 0.2
        cp = 1.0 / math.sqrt(self.dim + 2.0)
        d_param = 1.0 + self.dim / 2.0
        c_cov = 2.0 / (self.dim ** 2 + 6.0)

        while self.eval_count < self.budget:
            z = np.random.normal(0, 1, size=self.dim)
            step = A @ z
            cand = np.clip(curr_x + sigma * step, lb, ub)

            y = float(func(cand))
            self.eval_count += 1

            if y < best_y:
                best_y = y
                best_x = cand.copy()

            success = (y <= curr_y)
            p_succ = (1.0 - cp) * p_succ + cp * float(success)

            if success:
                curr_x = cand
                curr_y = y

                norm_z_sq = np.dot(z, z)
                if norm_z_sq > 1e-10:
                    factor = math.sqrt(1.0 - c_cov)
                    term = (math.sqrt(1.0 + (c_cov / (1.0 - c_cov)) * norm_z_sq) - 1.0) / norm_z_sq
                    Az = A @ z
                    A = factor * A + (factor * term) * np.outer(Az, z)

            # Step size adaptation (1/5th success rule)
            sigma = sigma * math.exp((p_succ - 0.2) / d_param)
            sigma = max(sigma, min_sigma)

            if sigma < min_sigma * 10 or np.max(np.abs(A)) > 1e10:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                sigma = 0.2
                A = np.diag(domain_range.copy())
                p_succ = 0.2

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
