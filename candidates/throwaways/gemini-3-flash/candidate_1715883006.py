# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A bounded Nelder-Mead simplex algorithm augmented with automated restarts upon simplex collapse or stagnation.
# Search state: Maintains a simplex of dim + 1 points and their objective values, alongside the global best solution.
# Candidate generation: Generates points via geometric simplex operations: reflection, expansion, contraction, and shrink.
# Selection and replacement: Replaces the worst simplex vertex with newly generated superior points according to Nelder-Mead rules.
# Adaptation: Simplex naturally adapts its geometry (stretching down slopes, contracting in wells) to match local contours.
# Exploration mechanisms: Re-initializes the simplex around the global best with randomized diversity when the volume collapses or progress halts.
# Exploitation mechanisms: Simplex contraction rapidly hones in on local minima once a basin is located.
# Boundary handling: All reflected and expanded vertices are clipped to remain inside the valid domain boundaries.
# Budget strategy: Simplex iterations evaluate trial points sequentially, strictly checking budget remaining before any evaluation.
# Closest known influences: Nelder-Mead simplex method.
# Novelty or unusual aspects: Combines classical derivative-free simplex transformations with robust multi-start re-seeding for global black-box domains.
# Failure modes: Can stall on noisy or highly discontinuous functions before triggering a restart.
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
        n_pts = self.dim + 1
        domain_range = ub - lb

        best_x = None
        best_y = float("inf")

        while self.eval_count < self.budget:
            # Initialize simplex
            simplex = np.zeros((n_pts, self.dim))
            fitness = np.zeros(n_pts)

            # First vertex is best known or random
            if best_x is not None:
                simplex[0] = best_x.copy()
            else:
                simplex[0] = np.random.uniform(lb, ub, size=self.dim)

            if self.eval_count < self.budget:
                fitness[0] = float(func(simplex[0]))
                self.eval_count += 1
                if fitness[0] < best_y:
                    best_y = fitness[0]
                    best_x = simplex[0].copy()

            # Remaining vertices created by perturbing along each dimension
            for i in range(1, n_pts):
                if self.eval_count >= self.budget:
                    break
                step = np.random.normal(0, 0.2, size=self.dim) * domain_range
                simplex[i] = np.clip(simplex[0] + step, lb, ub)
                y = float(func(simplex[i]))
                self.eval_count += 1
                fitness[i] = y
                if y < best_y:
                    best_y = y
                    best_x = simplex[i].copy()

            stagnation = 0
            max_stag = 15 * self.dim

            # Nelder-Mead parameters
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            while self.eval_count < self.budget and stagnation < max_stag:
                # Sort simplex
                idx = np.argsort(fitness)
                simplex = simplex[idx]
                fitness = fitness[idx]

                # Check simplex diameter for collapse
                diff = np.max(simplex) - np.min(simplex)
                if diff < 1e-7:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                yr = float(func(xr))
                self.eval_count += 1

                if yr < best_y:
                    best_y = yr
                    best_x = xr.copy()

                if fitness[0] <= yr < fitness[-2]:
                    simplex[-1] = xr
                    fitness[-1] = yr
                    stagnation = 0
                    continue

                if yr < fitness[0]:
                    # Expansion
                    if self.eval_count >= self.budget:
                        break
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    ye = float(func(xe))
                    self.eval_count += 1

                    if ye < best_y:
                        best_y = ye
                        best_x = xe.copy()

                    if ye < yr:
                        simplex[-1] = xe
                        fitness[-1] = ye
                    else:
                        simplex[-1] = xr
                        fitness[-1] = yr
                    stagnation = 0
                    continue

                # Contraction
                if yr < fitness[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)

                xc = np.clip(xc, lb, ub)
                if self.eval_count >= self.budget:
                    break
                yc = float(func(xc))
                self.eval_count += 1

                if yc < best_y:
                    best_y = yc
                    best_x = xc.copy()

                if yc < min(yr, fitness[-1]):
                    simplex[-1] = xc
                    fitness[-1] = yc
                    stagnation = 0
                else:
                    # Shrink
                    for j in range(1, n_pts):
                        if self.eval_count >= self.budget:
                            break
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        simplex[j] = np.clip(simplex[j], lb, ub)
                        yj = float(func(simplex[j]))
                        self.eval_count += 1
                        fitness[j] = yj
                        if yj < best_y:
                            best_y = yj
                            best_x = simplex[j].copy()
                    stagnation += 1

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
