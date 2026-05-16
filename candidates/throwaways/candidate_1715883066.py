# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Multi-Start Nelder-Mead simplex algorithm incorporating a Tabu archive to prevent re-optimizing discovered local minima.
# Search state: Retains current simplex vertices, objective values, Tabu local optima archive, and global optimum.
# Candidate generation: Generates geometric trial points via reflection, expansion, contraction, or shrinkage of the simplex.
# Selection and replacement: Replaces worst simplex vertex with successful geometric trials; re-seeds simplex upon diameter collapse.
# Adaptation: Simplex naturally deforms, elongates down valleys, and contracts around local optima without gradient evaluations.
# Exploration mechanisms: Re-seeding new simplex anchors at maximum distances from archived Tabu points enforces rigorous domain exploration.
# Exploitation mechanisms: Exact Nelder-Mead geometric contractions aggressively pinpoint the floor of the active local basin.
# Boundary handling: All geometric reflection and expansion candidate points are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates simplex transformations sequentially while strictly checking remaining evaluation budget.
# Closest known influences: Nelder-Mead Simplex / Multi-Start Optimization / Tabu Search.
# Novelty or unusual aspects: Combines exact geometric simplex deformation with distance-based Tabu filtering during restart anchor generation.
# Failure modes: Simplex can collapse into a degenerate flat sub-space if initial steps repeatedly hit active boundary constraints.
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

        best_x = None
        best_y = float("inf")
        tabu_optima = []
        tabu_radius = 1e-2 * math.sqrt(np.sum(domain_range ** 2))

        def create_simplex():
            # Generate initial anchor far from tabu list
            anchor = np.random.uniform(lb, ub, size=self.dim)
            if len(tabu_optima) > 0 and self.eval_count < self.budget:
                best_dist = 0.0
                for _ in range(10):
                    cand = np.random.uniform(lb, ub, size=self.dim)
                    d = np.min(np.linalg.norm(np.array(tabu_optima) - cand, axis=1))
                    if d > best_dist:
                        best_dist = d
                        anchor = cand
            
            s = np.zeros((self.dim + 1, self.dim))
            s[0] = anchor.copy()
            step = 0.1 * domain_range
            for i in range(self.dim):
                pt = anchor.copy()
                pt[i] = np.clip(pt[i] + step[i], lb[i], ub[i])
                s[i + 1] = pt
            return s

        simplex = create_simplex()
        fitness = np.full(self.dim + 1, float("inf"))

        for i in range(self.dim + 1):
            if self.eval_count >= self.budget:
                break
            y = float(func(simplex[i]))
            self.eval_count += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = simplex[i].copy()

        alpha, gamma_nm, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        stagnation = 0

        while self.eval_count < self.budget:
            sorted_idx = np.argsort(fitness)
            simplex = simplex[sorted_idx]
            fitness = fitness[sorted_idx]

            # Check diameter collapse
            diameter = np.max(np.linalg.norm(simplex[1:] - simplex[0], axis=1))
            if diameter < 1e-5 * math.sqrt(np.sum(domain_range ** 2)) or stagnation > 30:
                if len(tabu_optima) < 50:
                    tabu_optima.append(simplex[0].copy())
                
                simplex = create_simplex()
                fitness = np.full(self.dim + 1, float("inf"))
                for i in range(self.dim + 1):
                    if self.eval_count >= self.budget:
                        break
                    y = float(func(simplex[i]))
                    self.eval_count += 1
                    fitness[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = simplex[i].copy()
                stagnation = 0
                continue

            centroid = np.mean(simplex[:-1], axis=0)
            worst_x, worst_y = simplex[-1], fitness[-1]
            sec_worst_y = fitness[-2]
            best_curr_y = fitness[0]

            # Reflection
            ref_x = np.clip(centroid + alpha * (centroid - worst_x), lb, ub)
            if self.eval_count >= self.budget:
                break
            ref_y = float(func(ref_x))
            self.eval_count += 1

            if ref_y < best_y:
                best_y = ref_y
                best_x = ref_x.copy()

            if best_curr_y <= ref_y < sec_worst_y:
                simplex[-1] = ref_x
                fitness[-1] = ref_y
                stagnation = 0
                continue

            # Expansion
            if ref_y < best_curr_y:
                exp_x = np.clip(centroid + gamma_nm * (ref_x - centroid), lb, ub)
                if self.eval_count >= self.budget:
                    break
                exp_y = float(func(exp_x))
                self.eval_count += 1

                if exp_y < best_y:
                    best_y = exp_y
                    best_x = exp_x.copy()

                if exp_y < ref_y:
                    simplex[-1] = exp_x
                    fitness[-1] = exp_y
                else:
                    simplex[-1] = ref_x
                    fitness[-1] = ref_y
                stagnation = 0
                continue

            # Contraction
            if ref_y < worst_y:
                cont_x = np.clip(centroid + rho * (ref_x - centroid), lb, ub)
                if self.eval_count >= self.budget:
                    break
                cont_y = float(func(cont_x))
                self.eval_count += 1

                if cont_y < best_y:
                    best_y = cont_y
                    best_x = cont_x.copy()

                if cont_y <= ref_y:
                    simplex[-1] = cont_x
                    fitness[-1] = cont_y
                    stagnation = 0
                    continue
            else:
                cont_x = np.clip(centroid + rho * (worst_x - centroid), lb, ub)
                if self.eval_count >= self.budget:
                    break
                cont_y = float(func(cont_x))
                self.eval_count += 1

                if cont_y < best_y:
                    best_y = cont_y
                    best_x = cont_x.copy()

                if cont_y < worst_y:
                    simplex[-1] = cont_x
                    fitness[-1] = cont_y
                    stagnation = 0
                    continue

            # Shrink
            s0 = simplex[0]
            next_simp = np.zeros_like(simplex)
            next_simp[0] = s0.copy()
            next_fit = np.full(self.dim + 1, float("inf"))
            next_fit[0] = fitness[0]

            for i in range(1, self.dim + 1):
                if self.eval_count >= self.budget:
                    break
                shrink_x = np.clip(s0 + sigma_nm * (simplex[i] - s0), lb, ub)
                y = float(func(shrink_x))
                self.eval_count += 1
                next_simp[i] = shrink_x
                next_fit[i] = y

                if y < best_y:
                    best_y = y
                    best_x = shrink_x.copy()

            simplex = next_simp
            fitness = next_fit
            stagnation += 1

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
