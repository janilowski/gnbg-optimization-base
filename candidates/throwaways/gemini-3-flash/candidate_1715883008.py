# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A randomized directional hill climber based on Solis & Wets algorithm with multi-start capabilities.
# Search state: Retains the current solution, step size rho, search bias vector, stagnation counter, and global best.
# Candidate generation: Proposes positive and negative perturbation steps shifted by the directional bias vector.
# Selection and replacement: Evaluates positive step; if successful, accepts and updates bias. Otherwise evaluates reflection step.
# Adaptation: Expands step size upon successful moves, contracts upon failures, and dynamically shifts bias vector towards successful directions.
# Exploration mechanisms: Re-seeds the search at random domain locations whenever local step size collapses or stagnation threshold is breached.
# Exploitation mechanisms: Directional bias tracking acts as momentum, rapidly sliding down smooth gradients.
# Boundary handling: All test moves are clipped inside the variable bounds before evaluating function values.
# Budget strategy: Sequentially checks positive and negative probes while verifying evaluation budget availability.
# Closest known influences: Solis & Wets randomized hill climbing.
# Novelty or unusual aspects: Augments the classical single-point search with explicit bound tracking and multi-start automated resets.
# Failure modes: Can oscillate or stall in narrow curved valleys where isotropic steps fail to align with the valley axis.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

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

        rho = 0.2
        bias = np.zeros(self.dim)
        stagnation = 0
        max_stag = 70

        while self.eval_count < self.budget:
            # Generate step
            step = np.random.normal(0, 1, size=self.dim) * (rho * domain_range)
            
            # Positive probe
            trial_p = np.clip(curr_x + bias + step, lb, ub)
            y_p = float(func(trial_p))
            self.eval_count += 1

            if y_p < best_y:
                best_y = y_p
                best_x = trial_p.copy()

            if y_p < curr_y:
                curr_x = trial_p.copy()
                curr_y = y_p
                bias = 0.2 * bias + 0.4 * (bias + step)
                rho = min(1.2 * rho, 0.4)
                stagnation = 0
            else:
                if self.eval_count >= self.budget:
                    break

                # Negative probe
                trial_n = np.clip(curr_x - bias - step, lb, ub)
                y_n = float(func(trial_n))
                self.eval_count += 1

                if y_n < best_y:
                    best_y = y_n
                    best_x = trial_n.copy()

                if y_n < curr_y:
                    curr_x = trial_n.copy()
                    curr_y = y_n
                    bias = bias - 0.4 * (bias + step)
                    rho = min(1.2 * rho, 0.4)
                    stagnation = 0
                else:
                    bias = 0.5 * bias
                    rho = max(0.85 * rho, 1e-6)
                    stagnation += 1

            # Restart check
            if stagnation > max_stag or rho < 1e-5:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                rho = 0.2
                bias = np.zeros(self.dim)
                stagnation = 0

        return best_x, best_y
