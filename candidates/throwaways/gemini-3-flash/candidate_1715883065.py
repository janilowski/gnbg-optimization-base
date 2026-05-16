# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Variable Neighborhood Search (VNS) algorithm systematically expanding perturbation neighborhoods to escape local minima.
# Search state: Retains incumbent local optimum position, objective fitness value, active neighborhood index parameter, and global optimum.
# Candidate generation: Generates stochastic shaking perturbations within the active neighborhood radius, followed by intensive local Gaussian hill climbing.
# Selection and replacement: Replaces the incumbent local optimum if the local search following shaking discovers a superior solution basin.
# Adaptation: Systematically escalates neighborhood shaking radius upon local search failures and resets radius upon successful moves.
# Exploration mechanisms: Expanding neighborhood scales up to large fractions of the domain span ensures robust escape from deceptive attractor wells.
# Exploitation mechanisms: Intensive local hill climbing after each shaking step aggressively drives solutions down to the exact local minimum.
# Boundary handling: All shaking perturbations and local steps are explicitly clipped inside valid domain boundaries.
# Budget strategy: Allocates evaluation budgets sequentially across shaking and local search phases while checking remaining budget.
# Closest known influences: Variable Neighborhood Search VNS (Mladenovic & Hansen).
# Novelty or unusual aspects: Directly maps discrete VNS neighborhood transition ladders to exponential continuous Gaussian radius scales.
# Failure modes: Can experience sluggish progress if local search budgets are too small to resolve valleys before triggering neighborhood expansions.
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

        k_max = 5
        # scales: 0.02, 0.04, 0.08, 0.16, 0.32
        scales = [0.02 * (2 ** k) for k in range(k_max)]

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
        curr_x, curr_y = local_search(curr_x, curr_y, 80)

        k_neigh = 0
        stagnation = 0

        while self.eval_count < self.budget:
            # Shaking step in neighborhood k
            sigma_pert = scales[k_neigh]
            z_shaking = np.random.normal(0, 1, size=self.dim)
            shaking_x = np.clip(curr_x + z_shaking * (sigma_pert * domain_range), lb, ub)
            
            if self.eval_count >= self.budget:
                break
            shaking_y = float(func(shaking_x))
            self.eval_count += 1

            if shaking_y < best_y:
                best_y = shaking_y
                best_x = shaking_x.copy()

            # Local search
            cand_x, cand_y = local_search(shaking_x, shaking_y, 50)

            if cand_y < curr_y:
                curr_x = cand_x.copy()
                curr_y = cand_y
                k_neigh = 0  # Reset neighborhood
                stagnation = 0
            else:
                k_neigh = (k_neigh + 1) % k_max
                stagnation += 1

            if stagnation > 15:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                curr_x, curr_y = local_search(curr_x, curr_y, 80)
                k_neigh = 0
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
