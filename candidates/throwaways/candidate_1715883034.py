# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An Artificial Bee Colony (ABC) algorithm modeling employed, onlooker, and scout bees searching across food source solutions.
# Search state: Retains food source positions, objective values, trial stagnation counters, and global best solution.
# Candidate generation: Proposes new positions by perturbing a single random coordinate towards a randomly chosen distinct food source.
# Selection and replacement: Replaces existing food sources if trial positions achieve equal or superior fitness; abandoned sources are re-seeded randomly.
# Adaptation: Scout bees automatically reset food sources that exceed stagnation thresholds to maintain population diversity.
# Exploration mechanisms: Scout bee random resets and roulette wheel selection ensure active exploration across all domain regions.
# Exploitation mechanisms: Employed and onlooker bees concentrate search perturbations around successful food source solutions.
# Boundary handling: All food source modifications are strictly clipped inside valid domain boundaries.
# Budget strategy: Evaluates employed, onlooker, and scout phases sequentially while enforcing strict evaluation budget caps.
# Closest known influences: Artificial Bee Colony ABC (Karaboga).
# Novelty or unusual aspects: Employs dynamic scaling of roulette wheel selection probabilities to maintain selection pressure on negative or shifted benchmarks.
# Failure modes: Single-coordinate perturbations can struggle to make progress along highly diagonal non-separable valleys.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.n_sources = int(min(self.budget // 10, max(12, 2 * self.dim)))
        if self.n_sources > 50:
            self.n_sources = 50

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

        sources = np.random.uniform(lb, ub, size=(self.n_sources, self.dim))
        objective = np.full(self.n_sources, float("inf"))
        trials = np.zeros(self.n_sources, dtype=int)

        for i in range(self.n_sources):
            if self.eval_count >= self.budget:
                break
            y = float(func(sources[i]))
            self.eval_count += 1
            objective[i] = y
            if y < best_y:
                best_y = y
                best_x = sources[i].copy()

        stagnation_limit = max(10, self.dim)

        while self.eval_count < self.budget:
            # --- Employed Bee Phase ---
            for i in range(self.n_sources):
                if self.eval_count >= self.budget:
                    break

                k = np.random.randint(self.n_sources)
                while k == i:
                    k = np.random.randint(self.n_sources)

                j = np.random.randint(self.dim)
                phi = np.random.uniform(-1, 1)

                trial = sources[i].copy()
                trial[j] = np.clip(trial[j] + phi * (trial[j] - sources[k, j]), lb[j], ub[j])

                y = float(func(trial))
                self.eval_count += 1

                if y <= objective[i]:
                    objective[i] = y
                    sources[i] = trial
                    trials[i] = 0
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()
                else:
                    trials[i] += 1

            if self.eval_count >= self.budget:
                break

            # --- Calculate Fitness for Roulette Wheel ---
            # For minimization: shift objective values
            min_obj = np.min(objective)
            shifted = objective - min_obj
            fitness = 1.0 / (1.0 + shifted)
            probs = fitness / (np.sum(fitness) + 1e-12)

            # --- Onlooker Bee Phase ---
            t = 0
            i = 0
            while t < self.n_sources and self.eval_count < self.budget:
                if np.random.rand() < probs[i]:
                    t += 1
                    k = np.random.randint(self.n_sources)
                    while k == i:
                        k = np.random.randint(self.n_sources)

                    j = np.random.randint(self.dim)
                    phi = np.random.uniform(-1, 1)

                    trial = sources[i].copy()
                    trial[j] = np.clip(trial[j] + phi * (trial[j] - sources[k, j]), lb[j], ub[j])

                    y = float(func(trial))
                    self.eval_count += 1

                    if y <= objective[i]:
                        objective[i] = y
                        sources[i] = trial
                        trials[i] = 0
                        if y < best_y:
                            best_y = y
                            best_x = trial.copy()
                    else:
                        trials[i] += 1

                i = (i + 1) % self.n_sources

            # --- Scout Bee Phase ---
            if self.eval_count < self.budget:
                worst_trial_idx = np.argmax(trials)
                if trials[worst_trial_idx] > stagnation_limit:
                    sources[worst_trial_idx] = np.random.uniform(lb, ub, size=self.dim)
                    if self.eval_count < self.budget:
                        y = float(func(sources[worst_trial_idx]))
                        self.eval_count += 1
                        objective[worst_trial_idx] = y
                        trials[worst_trial_idx] = 0
                        if y < best_y:
                            best_y = y
                            best_x = sources[worst_trial_idx].copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
