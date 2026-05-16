# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Reptile Search Algorithm (RSA) modeling crocodile encircling, hunting coordination, and predatory lunging behaviors.
# Search state: Retains agent population positions, objective fitness values, average position vector, and global optimum.
# Candidate generation: Generates candidate moves across four distinct predatory phases shifting from high marching encircling to final lunges.
# Selection and replacement: Evaluated candidate positions directly replace prior agent coordinates; global best is updated upon discovering superior solutions.
# Adaptation: Evolutionary sense (ES) parameter decreases linearly over iterations to contract search radiuses during late hunting phases.
# Exploration mechanisms: High marching encircling and random dispersion factors in early phases maintain robust global domain exploration.
# Exploitation mechanisms: Final hunting lunges referenced directly against the global best optimum drive rapid convergence onto the elite basin.
# Boundary handling: All candidate agent positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Allocates search phases sequentially across quarters of the evaluation budget while checking remaining limits.
# Closest known influences: Reptile Search Algorithm RSA (Abualigah et al.).
# Novelty or unusual aspects: Vectorizes exact four-phase transition equations across coordinates to eliminate branching overhead.
# Failure modes: Multiplicative coordinate operations in Phase 2 and 3 can cause scaling instability if unclipped.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

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

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop[i]))
            self.eval_count += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        beta = 0.1

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            es = 2.0 * np.random.rand() * (1.0 - progress)
            pop_ave = np.mean(pop, axis=0)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                eta = best_x - pop_ave
                r_vec = np.random.rand(self.dim)
                perc_diff = np.abs(best_x - pop[i]) / (best_x + 1e-12)

                if progress < 0.25:
                    # Phase 1: High marching
                    step = best_x - eta * beta - r_vec * domain_range * 0.05
                    trial = pop[i] + step * 0.1
                elif progress < 0.5:
                    # Phase 2: Flank encircling
                    step = best_x * perc_diff * es * r_vec
                    trial = pop[i] + step
                elif progress < 0.75:
                    # Phase 3: Hunting coordination
                    trial = best_x * (pop_ave / (pop[i] + 1e-12)) * es
                else:
                    # Phase 4: Final lunge
                    trial = best_x - eta * 0.01 - pop[i] * es

                trial = np.clip(trial, lb, ub)
                y = float(func(trial))
                self.eval_count += 1
                pop[i] = trial

                if y < best_y:
                    best_y = y
                    best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
