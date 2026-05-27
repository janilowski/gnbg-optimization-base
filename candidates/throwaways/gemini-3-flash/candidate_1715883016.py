# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Grey Wolf Optimizer (GWO) algorithm guiding search trajectories using the top three elite leaders (alpha, beta, delta).
# Search state: Retains a population of wolf candidate positions, objective values, and the exact positions of alpha, beta, and delta leaders.
# Candidate generation: Updates positions by averaging the attraction steps towards alpha, beta, and delta leaders scaled by dynamic exploration vectors.
# Selection and replacement: Re-ranks population members each generation to update the alpha, beta, and delta leader assignments.
# Adaptation: Exploration parameter 'a' linearly decreases from 2.0 to 0.0 over the budget to transition from global dispersion to local convergence.
# Exploration mechanisms: Random vector coefficients C (scaling leader distance) and A (allowing steps away from leaders when |A| > 1) ensure global reach.
# Exploitation mechanisms: Attraction towards the centroid of the top three leaders converges the swarm around elite basins when |A| < 1.
# Boundary handling: All updated wolf positions are clipped to remain inside valid domain boundaries.
# Budget strategy: Evaluates population members sequentially in generational cycles until the exact evaluation budget is exhausted.
# Closest known influences: Grey Wolf Optimizer (Mirjalili).
# Novelty or unusual aspects: Combines classical GWO leadership mechanics with strict population bounding and robust numerical damping.
# Failure modes: Can experience premature stagnation if all three leaders collapse into the same deceptive local minimum.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 6, max(15, 2 * self.dim)))
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
        best_x = None
        best_y = float("inf")

        pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        fitness = np.full(self.pop_size, float("inf"))

        alpha_pos, alpha_y = np.zeros(self.dim), float("inf")
        beta_pos, beta_y = np.zeros(self.dim), float("inf")
        delta_pos, delta_y = np.zeros(self.dim), float("inf")

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop[i]))
            self.eval_count += 1
            fitness[i] = y

            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

            if y < alpha_y:
                delta_y, delta_pos = beta_y, beta_pos.copy()
                beta_y, beta_pos = alpha_y, alpha_pos.copy()
                alpha_y, alpha_pos = y, pop[i].copy()
            elif y < beta_y:
                delta_y, delta_pos = beta_y, beta_pos.copy()
                beta_y, beta_pos = y, pop[i].copy()
            elif y < delta_y:
                delta_y, delta_pos = y, pop[i].copy()

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            a = 2.0 * (1.0 - progress)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = np.abs(C1 * alpha_pos - pop[i])
                X1 = alpha_pos - A1 * D_alpha

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = np.abs(C2 * beta_pos - pop[i])
                X2 = beta_pos - A2 * D_beta

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = np.abs(C3 * delta_pos - pop[i])
                X3 = delta_pos - A3 * D_delta

                trial = (X1 + X2 + X3) / 3.0
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1
                pop[i] = trial
                fitness[i] = y

                if y < best_y:
                    best_y = y
                    best_x = trial.copy()

                if y < alpha_y:
                    delta_y, delta_pos = beta_y, beta_pos.copy()
                    beta_y, beta_pos = alpha_y, alpha_pos.copy()
                    alpha_y, alpha_pos = y, trial.copy()
                elif y < beta_y:
                    delta_y, delta_pos = beta_y, beta_pos.copy()
                    beta_y, beta_pos = y, trial.copy()
                elif y < delta_y:
                    delta_y, delta_pos = y, trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
