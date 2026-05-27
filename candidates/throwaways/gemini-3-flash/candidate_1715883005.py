# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A hybrid optimization method pairing Simulated Annealing with randomized coordinate descent steps.
# Search state: Retains the current solution and its objective value, current temperature, step size parameters, and global best.
# Candidate generation: Alternates between full-dimensional Gaussian perturbations and single-coordinate directional probes.
# Selection and replacement: Improving moves are always accepted; non-improving moves are accepted stochastically based on the Boltzmann probability.
# Adaptation: Temperature exponentially decays as the evaluation count increases, narrowing the acceptance probability for detrimental moves.
# Exploration mechanisms: High initial temperature permits overcoming local barriers and escaping deceptive basins early on.
# Exploitation mechanisms: Single-coordinate line probes at lower temperatures fine-tune the incumbent solution along individual axes.
# Boundary handling: All candidate solutions are constrained via clipping within domain boundaries.
# Budget strategy: Alternates search modes step-by-step while verifying remaining evaluation budget before every function call.
# Closest known influences: Simulated Annealing (Kirkpatrick) combined with random coordinate descent.
# Novelty or unusual aspects: Dynamically scales Boltzmann acceptance normalization using observed objective function variations.
# Failure modes: Can be inefficient on highly correlated landscapes where single-coordinate steps cannot make progress.
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

        temp = 100.0
        min_temp = 1e-4
        cooling_rate = (min_temp / temp) ** (1.0 / max(1, self.budget // 2))

        step_scale = 0.1
        est_scale = 1.0  # Estimated magnitude of function differences

        while self.eval_count < self.budget:
            mode = np.random.rand()
            
            if mode < 0.7:
                # Full Gaussian perturbation
                step = np.random.normal(0, 1, size=self.dim) * (step_scale * domain_range)
                trial = np.clip(curr_x + step, lb, ub)
            else:
                # Single coordinate perturbation
                idx = np.random.randint(self.dim)
                direction = np.random.choice([-1, 1])
                trial = curr_x.copy()
                trial[idx] = np.clip(trial[idx] + direction * (step_scale * domain_range[idx]), lb[idx], ub[idx])

            y = float(func(trial))
            self.eval_count += 1

            diff = y - curr_y

            # Update estimated scale of differences
            if abs(diff) > 1e-6:
                est_scale = 0.9 * est_scale + 0.1 * abs(diff)

            if diff < 0:
                # Improvement: accept
                curr_x = trial.copy()
                curr_y = y
                if y < best_y:
                    best_y = y
                    best_x = trial.copy()
            else:
                # Stochastic acceptance
                boltz_prob = math.exp(-diff / (temp * est_scale + 1e-12))
                if np.random.rand() < boltz_prob:
                    curr_x = trial.copy()
                    curr_y = y

            # Decay temperature and step size
            temp = max(min_temp, temp * cooling_rate)
            step_scale = max(0.001, step_scale * 0.999)

        return best_x, best_y
