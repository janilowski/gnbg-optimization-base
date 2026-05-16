# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Compact Genetic Algorithm (cGA) maintaining a virtual population distribution via pairwise duels.
# Search state: Retains current mean vector, variance vector, virtual population size parameter, and global optimum.
# Candidate generation: Generates two competing candidate points per iteration via independent Gaussian sampling around the mean.
# Selection and replacement: Compares the two candidate solutions to identify the superior winner and inferior loser.
# Adaptation: Shifts the distribution mean towards the winner and away from the loser while adjusting coordinate variances based on duel divergence.
# Exploration mechanisms: Virtual population inertia prevents instantaneous collapse of coordinate standard deviations.
# Exploitation mechanisms: Continual updates favoring winning positions over losing positions steadily pull the distribution towards optimal basins.
# Boundary handling: All sampled candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Allocates evaluations in precise pairs per iteration while strictly monitoring remaining evaluation budget.
# Closest known influences: Compact Genetic Algorithm cGA (Harik et al.).
# Novelty or unusual aspects: Extends discrete probability vector updates to real-valued continuous moments without storing actual population archives.
# Failure modes: Can experience false convergence if early pairwise duels repeatedly sample deceptive local attractors.
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

        best_x = None
        best_y = float("inf")

        mean = lb + 0.5 * domain_range
        var = (0.25 * domain_range) ** 2
        min_var = (1e-6 * domain_range) ** 2

        virtual_pop = 50.0
        step_size = 1.0 / virtual_pop

        while self.eval_count < self.budget - 1:
            std = np.sqrt(var)
            
            # Generate two competing individuals
            z1 = np.random.normal(0, 1, size=self.dim)
            x1 = np.clip(mean + z1 * std, lb, ub)
            y1 = float(func(x1))
            self.eval_count += 1

            if y1 < best_y:
                best_y = y1
                best_x = x1.copy()

            z2 = np.random.normal(0, 1, size=self.dim)
            x2 = np.clip(mean + z2 * std, lb, ub)
            y2 = float(func(x2))
            self.eval_count += 1

            if y2 < best_y:
                best_y = y2
                best_x = x2.copy()

            # Determine winner and loser
            if y1 < y2:
                winner, loser = x1, x2
            else:
                winner, loser = x2, x1

            # Update mean and variance
            mean_diff = winner - loser
            mean = mean + step_size * mean_diff

            var_update = ((winner - mean) ** 2) - ((loser - mean) ** 2)
            var = var + step_size * var_update
            var = np.maximum(var, min_var)

            # Check if variance collapsed
            if np.max(np.sqrt(var) / domain_range) < 1e-5:
                mean = np.random.uniform(lb, ub, size=self.dim)
                var = (0.25 * domain_range) ** 2

        # If 1 evaluation left
        if self.eval_count < self.budget:
            std = np.sqrt(var)
            x1 = np.clip(mean + np.random.normal(0, 1, size=self.dim) * std, lb, ub)
            y1 = float(func(x1))
            self.eval_count += 1
            if y1 < best_y:
                best_y = y1
                best_x = x1.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
