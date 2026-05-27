# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A bi-population inspired multi-start hill climbing method switching between large exploratory regimes and small intensive local regimes.
# Search state: Maintains an elite archive of the best discovered optima across restarts, current search position, and active regime state.
# Candidate generation: Proposes Gaussian perturbations scaled by regime-specific step sizes around either random global seeds or archive elites.
# Selection and replacement: Uses greedy acceptance within each local hill climbing run; successful local optima are stored in the elite archive.
# Adaptation: Alternates budget allocation between wide global exploration (large steps) and deep local exploitation (small steps).
# Exploration mechanisms: Interleaves random global restarts and wide-radius sampling to discover disparate attraction basins.
# Exploitation mechanisms: Small-step local hill climbing refines known elite archive positions to high precision.
# Boundary handling: All candidate solutions are clipped strictly inside domain bounds.
# Budget strategy: Allocates discrete blocks of evaluation budget to alternating regimes, stopping immediately when the budget limit is reached.
# Closest known influences: BIPOP-CMA-ES restart regime structure.
# Novelty or unusual aspects: Combines BIPOP scheduling with lightweight randomized hill climbing instead of full covariance matrix updates.
# Failure modes: Can waste budget in exploratory regimes if the landscape is unimodal and smooth.
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

        archive_x = []
        archive_y = []

        while self.eval_count < self.budget:
            # Decide regime: 0 for large global, 1 for small local around elite
            if len(archive_x) == 0 or np.random.rand() < 0.6:
                # Global regime
                init_samples = min(10, self.budget - self.eval_count)
                if init_samples <= 0:
                    break

                cands = np.random.uniform(lb, ub, size=(init_samples, self.dim))
                vals = np.zeros(init_samples)
                for i in range(init_samples):
                    if self.eval_count >= self.budget:
                        vals[i] = float("inf")
                        continue
                    y = float(func(cands[i]))
                    self.eval_count += 1
                    vals[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = cands[i].copy()

                best_idx = np.argmin(vals)
                curr_x = cands[best_idx].copy()
                curr_y = vals[best_idx]
                sigma = 0.25
                max_steps = 80
            else:
                # Local regime around elite
                elite_idx = np.random.randint(len(archive_x))
                curr_x = archive_x[elite_idx].copy()
                curr_y = archive_y[elite_idx]
                sigma = 0.05
                max_steps = 40

            step_count = 0
            stagnation = 0

            while self.eval_count < self.budget and step_count < max_steps and stagnation < 20:
                step = np.random.normal(0, 1, size=self.dim) * (sigma * domain_range)
                trial = np.clip(curr_x + step, lb, ub)
                y = float(func(trial))
                self.eval_count += 1
                step_count += 1

                if y < best_y:
                    best_y = y
                    best_x = trial.copy()

                if y < curr_y:
                    curr_x = trial.copy()
                    curr_y = y
                    stagnation = 0
                else:
                    sigma *= 0.9
                    stagnation += 1

            # Save local optimum to archive
            if len(archive_x) < 10:
                archive_x.append(curr_x.copy())
                archive_y.append(curr_y)
            else:
                worst_idx = np.argmax(archive_y)
                if curr_y < archive_y[worst_idx]:
                    archive_x[worst_idx] = curr_x.copy()
                    archive_y[worst_idx] = curr_y

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
