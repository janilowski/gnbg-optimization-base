# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous random search algorithm utilizing heavy-tailed Lévy flight perturbations with adaptive step scaling.
# Search state: Retains incumbent solution position, objective fitness value, step size vector, and global optimum.
# Candidate generation: Proposes candidate neighborhood batches via Lévy flight jumps generated using Mantegna's numerical approximation.
# Selection and replacement: Moves to the superior neighborhood candidate if objective fitness improves upon the incumbent.
# Adaptation: Multiplicatively expands step size upon successful jumps and contracts step size upon neighborhood search failures.
# Exploration mechanisms: Occasional extremely large heavy-tailed Lévy jumps prevent entrapment in local attraction basins.
# Exploitation mechanisms: Adaptive step size contraction during stagnation phases refines the incumbent solution locally.
# Boundary handling: All Lévy jump candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates neighborhood candidate batches sequentially while strictly monitoring remaining evaluation budget.
# Closest known influences: Lévy Flight Random Search (Pavlyukevich).
# Novelty or unusual aspects: Pre-computes exact gamma constants for robust Mantegna step generation across arbitrary domain dimensions.
# Failure modes: Can exhibit stochastic oscillation on smooth unimodal surfaces due to unconstrained heavy-tailed jumps.
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

        sigma = 0.05 * domain_range
        min_sigma = 1e-6 * domain_range
        max_sigma = 0.25 * domain_range

        stagnation = 0
        batch_size = min(10, max(4, self.dim))

        beta = 1.5
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                   (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)

        while self.eval_count < self.budget:
            best_neigh_x = None
            best_neigh_y = float("inf")

            for _ in range(batch_size):
                if self.eval_count >= self.budget:
                    break

                # Mantegna algorithm for Lévy flight
                u = np.random.normal(0, sigma_u, size=self.dim)
                v = np.random.normal(0, 1, size=self.dim)
                levy_step = u / (np.abs(v) ** (1.0 / beta))

                trial = np.clip(curr_x + levy_step * sigma, lb, ub)
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
                sigma = np.maximum(sigma * 0.9, min_sigma)
                stagnation += 1

            if stagnation > 25 or np.max(sigma / domain_range) < 1e-5:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                sigma = 0.05 * domain_range
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
