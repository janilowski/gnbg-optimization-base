# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Simulated Annealing algorithm screening candidate mutation pools using a lightweight quadratic regression surrogate.
# Search state: Stores incumbent solution position, objective fitness value, current temperature, step size, regression buffer, and global optimum.
# Candidate generation: Generates candidate pools via Gaussian mutation around the incumbent, screening them with a fitted least-squares quadratic surrogate.
# Selection and replacement: Evaluates the single most promising predicted candidate on the true objective; accepts moves via Boltzmann probability.
# Adaptation: Exponential temperature cooling reduces acceptance probability for inferior moves; step size adapts based on acceptance rates.
# Exploration mechanisms: Boltzmann acceptance and stochastic candidate pool generation prevent trapping in shallow local minima.
# Exploitation mechanisms: Least-squares quadratic surrogate screening accurately identifies descent trajectories in smooth local basins.
# Boundary handling: All candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates exactly one true objective call per surrogate filtering cycle until budget exhaustion.
# Closest known influences: Surrogate-Assisted Optimization / Simulated Annealing.
# Novelty or unusual aspects: Embeds lightweight diagonal quadratic least-squares regression directly into the Simulated Annealing proposal filter.
# Failure modes: Least-squares regression can become ill-conditioned if local buffer points cluster tightly on degenerate linear planes.
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

        T_start = 1.0
        T_end = 0.001
        alpha_cool = (T_end / T_start) ** (1.0 / (self.budget - 1.0 + 1e-12))
        T = T_start

        buffer_x = [curr_x.copy()]
        buffer_y = [curr_y]
        max_buffer = max(15, 2 * self.dim + 5)

        delta_norm = abs(curr_y) + 1.0
        stagnation = 0

        while self.eval_count < self.budget:
            n_pool = min(10, max(4, self.dim // 2))
            pool_x = np.zeros((n_pool, self.dim))
            for i in range(n_pool):
                z = np.random.normal(0, 1, size=self.dim)
                pool_x[i] = np.clip(curr_x + z * sigma, lb, ub)

            chosen_idx = 0
            if len(buffer_x) >= 2 * self.dim + 1:
                # Fit diagonal quadratic surrogate: y = c + sum b_i x_i + sum a_i x_i^2
                X_mat = np.array(buffer_x)
                Y_mat = np.array(buffer_y)

                # Normalize features for stability
                mu = np.mean(X_mat, axis=0)
                std = np.std(X_mat, axis=0) + 1e-8
                X_norm = (X_mat - mu) / std

                # Features: 1, x_1..x_D, x_1^2..x_D^2 (1 + 2*D columns)
                Phi = np.hstack((np.ones((len(buffer_x), 1)), X_norm, X_norm ** 2))
                
                try:
                    params, _, _, _ = np.linalg.lstsq(Phi, Y_mat, rcond=None)
                    
                    pool_norm = (pool_x - mu) / std
                    Phi_pool = np.hstack((np.ones((n_pool, 1)), pool_norm, pool_norm ** 2))
                    pred_y = Phi_pool @ params
                    chosen_idx = np.argmin(pred_y)
                except np.linalg.LinAlgError:
                    chosen_idx = np.random.randint(n_pool)
            else:
                chosen_idx = np.random.randint(n_pool)

            cand = pool_x[chosen_idx]
            y = float(func(cand))
            self.eval_count += 1

            if len(buffer_x) >= max_buffer:
                buffer_x.pop(0)
                buffer_y.pop(0)

            buffer_x.append(cand.copy())
            buffer_y.append(y)

            if y < best_y:
                best_y = y
                best_x = cand.copy()
                delta_norm = 0.5 * delta_norm + 0.5 * abs(y - curr_y)

            delta = y - curr_y

            if delta < 0 or np.random.rand() < math.exp(-delta / (T * delta_norm + 1e-12)):
                curr_x = cand
                curr_y = y
                sigma = np.minimum(sigma * 1.05, max_sigma)
                stagnation = 0
            else:
                sigma = np.maximum(sigma * 0.95, min_sigma)
                stagnation += 1

            T *= alpha_cool

            if stagnation > 50 or np.max(sigma / domain_range) < 1e-5:
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
