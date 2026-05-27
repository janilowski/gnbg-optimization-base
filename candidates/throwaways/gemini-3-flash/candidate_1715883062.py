# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Simulated Annealing algorithm incorporating physical momentum in candidate proposal steps.
# Search state: Retains incumbent solution position, objective fitness value, momentum velocity vector, current temperature, and global optimum.
# Candidate generation: Proposes candidate points via physical velocity vectors combining past momentum with Gaussian stochastic perturbations.
# Selection and replacement: Accepts improving moves unconditionally; accepts inferior moves with Boltzmann probability scaled by current temperature.
# Adaptation: Exponential cooling schedule steadily reduces acceptance probability for inferior moves as search progresses.
# Exploration mechanisms: Boltzmann acceptance and physical momentum allow trajectories to climb out of local attraction wells.
# Exploitation mechanisms: Low temperatures in late iterations freeze out inferior transitions, forcing greedy local convergence.
# Boundary handling: All candidate positions are clipped inside domain bounds; velocity vectors are inverted upon boundary collisions.
# Budget strategy: Evaluates proposed momentum steps sequentially while strictly checking remaining evaluation budget.
# Closest known influences: Simulated Annealing SA (Kirkpatrick et al.) / Momentum Gradient Descent.
# Novelty or unusual aspects: Augments standard Markov chain SA perturbations with continuous momentum velocity tracking.
# Failure modes: Can experience slow convergence if initial temperature or step magnitude parameters are poorly calibrated to the landscape scale.
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

        vel = np.zeros(self.dim)
        momentum = 0.5
        sigma = 0.1 * domain_range
        min_sigma = 1e-6 * domain_range

        T_start = 1.0
        T_end = 0.001
        alpha_cool = (T_end / T_start) ** (1.0 / (self.budget - 1.0 + 1e-12))
        T = T_start

        # Estimate scale delta
        delta_norm = abs(curr_y) + 1.0

        stagnation = 0

        while self.eval_count < self.budget:
            step = np.random.normal(0, 1, size=self.dim) * sigma
            v_new = momentum * vel + step
            cand = curr_x + v_new

            # Boundary handling
            out = (cand < lb) | (cand > ub)
            v_new[out] = -0.5 * v_new[out]
            cand = np.clip(cand, lb, ub)

            y = float(func(cand))
            self.eval_count += 1

            if y < best_y:
                best_y = y
                best_x = cand.copy()
                delta_norm = 0.5 * delta_norm + 0.5 * abs(y - curr_y)

            delta = y - curr_y

            # Acceptance check
            if delta < 0 or np.random.rand() < math.exp(-delta / (T * delta_norm + 1e-12)):
                curr_x = cand
                curr_y = y
                vel = v_new
                stagnation = 0
                sigma = np.minimum(sigma * 1.01, 0.3 * domain_range)
            else:
                vel = -0.5 * vel
                stagnation += 1
                sigma = np.maximum(sigma * 0.99, min_sigma)

            T *= alpha_cool

            if stagnation > 50:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                vel = np.zeros(self.dim)
                sigma = 0.1 * domain_range
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
