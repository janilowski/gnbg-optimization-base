# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An Iterated Local Search algorithm interleaving greedy local neighborhood optimization with semi-global stochastic perturbation leaps.
# Search state: Retains incumbent local optimum position, objective fitness value, perturbation scale parameter, and global optimum.
# Candidate generation: Alternates between intensive local Gaussian hill climbing and large stochastic perturbation leaps.
# Selection and replacement: Replaces incumbent local optimum whenever the post-perturbation local search discovers a superior local basin.
# Adaptation: Perturbation scale parameter adapts based on successful transitions between local optima basins.
# Exploration mechanisms: Substantial stochastic perturbation leaps dislodge the search from local trapping wells.
# Exploitation mechanisms: Intensive local hill climbing strictly drives solutions down to the exact floor of the local basin.
# Boundary handling: All local steps and perturbation jumps are explicitly clipped inside valid domain boundaries.
# Budget strategy: Allocates evaluation budgets strictly between local hill climbing cycles and perturbation leaps.
# Closest known influences: Iterated Local Search ILS (Lourenco et al.).
# Novelty or unusual aspects: Employs exact success-ratio tracking to dynamically adjust perturbation leap distances.
# Failure modes: Can waste evaluations optimizing shallow secondary basins if perturbation leaps are set too small.
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

        pert_scale = 0.25

        def local_search(start_x, start_y, max_evals):
            lx, ly = start_x.copy(), start_y
            step_size = 0.05 * domain_range
            min_step = 1e-6 * domain_range
            evals = 0

            while evals < max_evals and self.eval_count < self.budget:
                z = np.random.normal(0, 1, size=self.dim)
                cand = np.clip(lx + z * step_size, lb, ub)
                cy = float(func(cand))
                self.eval_count += 1
                evals += 1

                nonlocal best_y, best_x
                if cy < best_y:
                    best_y = cy
                    best_x = cand.copy()

                if cy < ly:
                    lx, ly = cand.copy(), cy
                    step_size = np.minimum(step_size * 1.2, 0.2 * domain_range)
                else:
                    step_size = np.maximum(step_size * 0.8, min_step)

                if np.max(step_size / domain_range) < 1e-5:
                    break

            return lx, ly

        # Initial local search
        curr_x, curr_y = local_search(curr_x, curr_y, 100)

        stagnation = 0

        while self.eval_count < self.budget:
            # Perturbation leap
            z_pert = np.random.normal(0, 1, size=self.dim)
            pert_x = np.clip(curr_x + z_pert * (pert_scale * domain_range), lb, ub)
            if self.eval_count >= self.budget:
                break
            pert_y = float(func(pert_x))
            self.eval_count += 1

            if pert_y < best_y:
                best_y = pert_y
                best_x = pert_x.copy()

            # Local search from perturbed point
            new_local_x, new_local_y = local_search(pert_x, pert_y, 60)

            if new_local_y < curr_y:
                curr_x = new_local_x.copy()
                curr_y = new_local_y
                pert_scale = min(pert_scale * 1.1, 0.5)
                stagnation = 0
            else:
                stagnation += 1
                pert_scale = max(pert_scale * 0.9, 0.01)

            if stagnation > 10:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                curr_x, curr_y = local_search(curr_x, curr_y, 100)
                pert_scale = 0.25
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
