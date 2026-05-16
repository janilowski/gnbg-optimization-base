# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Pattern Search algorithm integrating a Tabu list archive to forbid redundant evaluations around previously visited points.
# Search state: Retains incumbent solution position, objective fitness value, current step size parameter, Tabu list archive, and global optimum.
# Candidate generation: Generates coordinate axial test steps around the incumbent solution, filtering out points inside Tabu exclusion zones.
# Selection and replacement: Moves to the best feasible axial step if it improves upon the incumbent; archives the abandoned position into the Tabu list.
# Adaptation: Multiplicatively expands step size upon successful moves and halves step size upon local neighborhood failures.
# Exploration mechanisms: Tabu list repulsion prevents cyclic entrapment and forces search vectors into unvisited domain sectors.
# Exploitation mechanisms: Exact coordinate axial steps and step size contractions rapidly pinpoint local valley minima.
# Boundary handling: All axial test points are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates non-Tabu axial points sequentially while rigorously checking remaining evaluation budget.
# Closest known influences: Pattern Search / Tabu Search continuous (Glover).
# Novelty or unusual aspects: Directly embeds continuous distance-based Tabu memory filtering into coordinate pattern search probes.
# Failure modes: High memory comparisons against Tabu archives can become computationally burdensome for extremely large archives in high dimensions.
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

        sigma = 0.1
        min_sigma = 1e-6
        max_sigma = 0.5

        tabu_list = []
        max_tabu = 50
        tabu_radius = 1e-3 * math.sqrt(np.sum(domain_range ** 2))

        stagnation = 0

        while self.eval_count < self.budget:
            candidates = []

            # Axial steps: + and - for each dimension
            for i in range(self.dim):
                step = np.zeros(self.dim)
                step[i] = sigma * domain_range[i]
                
                c1 = np.clip(curr_x + step, lb, ub)
                c2 = np.clip(curr_x - step, lb, ub)

                candidates.extend([c1, c2])

            # Filter out tabu points
            feasible_cands = []
            if len(tabu_list) > 0:
                T_mat = np.array(tabu_list)
                for cand in candidates:
                    dists = np.linalg.norm(T_mat - cand, axis=1)
                    if np.min(dists) > tabu_radius:
                        feasible_cands.append(cand)
            else:
                feasible_cands = candidates

            if len(feasible_cands) == 0:
                # All candidates were tabu, force random jump
                jump = np.random.uniform(lb, ub, size=self.dim)
                feasible_cands.append(jump)

            best_neigh_x = None
            best_neigh_y = float("inf")

            for cand in feasible_cands:
                if self.eval_count >= self.budget:
                    break
                y = float(func(cand))
                self.eval_count += 1

                if y < best_neigh_y:
                    best_neigh_y = y
                    best_neigh_x = cand.copy()

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            if self.eval_count >= self.budget:
                break

            if best_neigh_y < curr_y:
                # Successful move
                if len(tabu_list) >= max_tabu:
                    tabu_list.pop(0)
                tabu_list.append(curr_x.copy())

                curr_x = best_neigh_x.copy()
                curr_y = best_neigh_y
                sigma = min(sigma * 1.2, max_sigma)
                stagnation = 0
            else:
                sigma = max(sigma * 0.5, min_sigma)
                stagnation += 1

            if stagnation > 15 or sigma < min_sigma * 2:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                sigma = 0.1
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
