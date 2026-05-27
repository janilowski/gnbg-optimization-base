# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Harris Hawks Optimization (HHO) algorithm modeling cooperative chasing, surprise pounces, and multi-stage rapid dive tactics.
# Search state: Retains agent population positions, objective fitness values, average swarm position, and global optimum.
# Candidate generation: Switches between wide exploration leaps and four distinct attacking modes (soft/hard besiege with or without rapid dives).
# Selection and replacement: Evaluated candidate positions replace prior agent coordinates if objective fitness improves; updates global best optimum.
# Adaptation: Escaping energy parameter E contracts linearly and oscillates stochastically to transition swarm from exploration to exploitation.
# Exploration mechanisms: Referencing moves against random swarm members or domain bounds maintains strong unconstrained exploration when |E| >= 1.
# Exploitation mechanisms: Soft and hard besiege maneuvers contract agent trajectories tightly around the escaping prey (global best).
# Boundary handling: All agent candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population agents sequentially in generational iterations while strictly checking remaining evaluation budget.
# Closest known influences: Harris Hawks Optimization HHO (Heidari et al.).
# Novelty or unusual aspects: Directly embeds Levy flight rapid dive tests into the single-agent evaluation loop to escape local valleys.
# Failure modes: Evaluating multiple trial dives per agent during Phase 3 and 4 reduces total generational iterations under tight budgets.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 10, max(20, 2 * self.dim)))
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
            pop_ave = np.mean(pop, axis=0)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                e0 = 2.0 * np.random.rand() - 1.0
                e_param = 2.0 * e0 * (1.0 - progress)
                abs_e = abs(e_param)

                if abs_e >= 1.0:
                    # Exploration
                    q_rand = np.random.rand()
                    if q_rand >= 0.5:
                        r_idx = np.random.randint(self.pop_size)
                        r1, r2 = np.random.rand(), np.random.rand()
                        trial = pop[r_idx] - r1 * np.abs(pop[r_idx] - 2.0 * r2 * pop[i])
                    else:
                        r3, r4 = np.random.rand(), np.random.rand()
                        step_rand = lb + r4 * domain_range
                        trial = (best_x - pop_ave) - r3 * step_rand
                    trial = np.clip(trial, lb, ub)
                    y = float(func(trial))
                    self.eval_count += 1
                    if y < fitness[i]:
                        pop[i] = trial
                        fitness[i] = y
                        if y < best_y:
                            best_y = y
                            best_x = trial.copy()
                else:
                    # Exploitation
                    r_chance = np.random.rand()
                    j_jump = 2.0 * (1.0 - np.random.rand())

                    if r_chance >= 0.5 and abs_e >= 0.5:
                        # Soft besiege
                        delta_x = best_x - pop[i]
                        trial = delta_x - e_param * np.abs(j_jump * best_x - pop[i])
                        trial = np.clip(trial, lb, ub)
                        y = float(func(trial))
                        self.eval_count += 1
                        if y < fitness[i]:
                            pop[i] = trial
                            fitness[i] = y
                            if y < best_y:
                                best_y = y
                                best_x = trial.copy()

                    elif r_chance >= 0.5 and abs_e < 0.5:
                        # Hard besiege
                        delta_x = best_x - pop[i]
                        trial = best_x - e_param * np.abs(delta_x)
                        trial = np.clip(trial, lb, ub)
                        y = float(func(trial))
                        self.eval_count += 1
                        if y < fitness[i]:
                            pop[i] = trial
                            fitness[i] = y
                            if y < best_y:
                                best_y = y
                                best_x = trial.copy()

                    elif r_chance < 0.5 and abs_e >= 0.5:
                        # Soft besiege with progressive rapid dives
                        y_cand = best_x - e_param * np.abs(j_jump * best_x - pop[i])
                        y_cand = np.clip(y_cand, lb, ub)
                        fit_y = float(func(y_cand))
                        self.eval_count += 1

                        if fit_y < fitness[i]:
                            pop[i] = y_cand
                            fitness[i] = fit_y
                            if fit_y < best_y:
                                best_y = fit_y
                                best_x = y_cand.copy()
                        else:
                            if self.eval_count >= self.budget:
                                break
                            z_step = levy_step(self.dim)
                            z_cand = y_cand + z_step * 0.05 * domain_range
                            z_cand = np.clip(z_cand, lb, ub)
                            fit_z = float(func(z_cand))
                            self.eval_count += 1

                            if fit_z < fitness[i]:
                                pop[i] = z_cand
                                fitness[i] = fit_z
                                if fit_z < best_y:
                                    best_y = fit_z
                                    best_x = z_cand.copy()

                    elif r_chance < 0.5 and abs_e < 0.5:
                        # Hard besiege with progressive rapid dives
                        y_cand = best_x - e_param * np.abs(j_jump * best_x - pop_ave)
                        y_cand = np.clip(y_cand, lb, ub)
                        fit_y = float(func(y_cand))
                        self.eval_count += 1

                        if fit_y < fitness[i]:
                            pop[i] = y_cand
                            fitness[i] = fit_y
                            if fit_y < best_y:
                                best_y = fit_y
                                best_x = y_cand.copy()
                        else:
                            if self.eval_count >= self.budget:
                                break
                            z_step = levy_step(self.dim)
                            z_cand = y_cand + z_step * 0.05 * domain_range
                            z_cand = np.clip(z_cand, lb, ub)
                            fit_z = float(func(z_cand))
                            self.eval_count += 1

                            if fit_z < fitness[i]:
                                pop[i] = z_cand
                                fitness[i] = fit_z
                                if fit_z < best_y:
                                    best_y = fit_z
                                    best_x = z_cand.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
