# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Marine Predators Algorithm (MPA) mimicking optimal foraging strategies across Brownian and Levy flight motion regimes.
# Search state: Retains agent population positions, objective fitness values, elite matrix of global optimum copies, and global optimum.
# Candidate generation: Proposes positions across three distinct evaluation phases transitioning from Brownian diffusive steps to long-tailed Levy flights.
# Selection and replacement: Evaluated candidate positions replace prior parent coordinates if objective fitness improves; updates elite matrix upon discovering superior solutions.
# Adaptation: Automatically shifts motion regimes from pure Brownian exploration to hybrid predator-prey dynamics and late Levy flight exploitation.
# Exploration mechanisms: Fish Aggregating Device (FAD) stochastic jumps and Levy flight long tails dislodge agents from local trapping basins.
# Exploitation mechanisms: Centering elite matrix rows directly on the global optimum pulls agents into the leading optimum basin.
# Boundary handling: All agent candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Allocates search regimes precisely across thirds of the total evaluation budget while checking remaining limits.
# Closest known influences: Marine Predators Algorithm MPA (Faramarzi et al.).
# Novelty or unusual aspects: Pre-computes vectorized Levy flight distributions using exact inverse transform gamma scaling.
# Failure modes: Stochastic FAD jumps can cause late-stage jitter if jump probability parameters are set too aggressively.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 8, max(20, 2 * self.dim)))
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

        fad_prob = 0.2
        p_param = 0.5
        half_pop = self.pop_size // 2

        def levy_step(shape):
            beta = 1.5
            sigma = (math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0) /
                     (math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0)))) ** (1.0 / beta)
            u = np.random.normal(0, sigma, size=shape)
            v = np.random.normal(0, 1, size=shape)
            step = u / (np.abs(v) ** (1.0 / beta) + 1e-12)
            return step

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            next_pop = np.zeros_like(pop)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                r_vec = np.random.rand(self.dim)

                if progress < 0.333:
                    # Phase 1: High velocity ratio (Brownian motion)
                    r_b = np.random.normal(0, 1, size=self.dim)
                    step = r_b * (best_x - r_b * pop[i])
                    trial = pop[i] + p_param * r_vec * step
                elif progress < 0.667:
                    # Phase 2: Unit velocity ratio
                    if i < half_pop:
                        r_l = levy_step(self.dim)
                        step = r_l * (best_x - r_l * pop[i])
                        trial = pop[i] + p_param * r_vec * step
                    else:
                        r_b = np.random.normal(0, 1, size=self.dim)
                        step = r_b * (r_b * best_x - pop[i])
                        trial = best_x + p_param * p_param * step
                else:
                    # Phase 3: Low velocity ratio (Levy flight)
                    r_l = levy_step(self.dim)
                    step = r_l * (r_l * best_x - pop[i])
                    trial = best_x + p_param * p_param * step

                # Fish Aggregating Devices (FADs) effect
                if np.random.rand() < fad_prob:
                    u_prob = np.random.rand(self.dim) < fad_prob
                    r1, r2 = np.random.choice(self.pop_size, size=2, replace=False)
                    trial = np.where(u_prob, trial + np.random.rand(self.dim) * (pop[r1] - pop[r2]), trial)

                trial = np.clip(trial, lb, ub)
                y = float(func(trial))
                self.eval_count += 1

                if y < fitness[i]:
                    pop[i] = trial
                    fitness[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
