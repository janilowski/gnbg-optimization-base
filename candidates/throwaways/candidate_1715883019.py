# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Tabu Search algorithm maintaining an archive of penalized sub-regions to prevent cyclic entrapment.
# Search state: Retains incumbent position, best objective value, a Tabu list of local minima coordinates, and stagnation counters.
# Candidate generation: Proposes Gaussian perturbations around the incumbent; candidate points falling inside Tabu radii are rejected or penalized.
# Selection and replacement: Evaluates valid neighborhood candidates and moves to the superior neighbor; updates Tabu list upon local stagnation.
# Adaptation: Step size contracts during local exploitation phases and resets upon triggering Tabu avoidance restarts.
# Exploration mechanisms: Tabu archive explicitly repels search trajectories from re-visiting explored basins, forcing global dispersion.
# Exploitation mechanisms: Small-step Gaussian neighborhood sampling fine-tunes the incumbent until stagnation triggers Tabu archival.
# Boundary handling: All neighborhood candidate points are strictly clipped inside valid domain boundaries.
# Budget strategy: Evaluates candidate neighborhood batches sequentially while rigorously checking remaining budget.
# Closest known influences: Continuous Tabu Search (Glover).
# Novelty or unusual aspects: Employs Euclidean distance thresholding in continuous spaces to dynamically mask trapped local attraction wells.
# Failure modes: Can become computationally expensive if Tabu list size grows large in high-dimensional spaces.
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
        tabu_radius = 0.05 * math.sqrt(np.sum(domain_range ** 2))

        curr_x = np.random.uniform(lb, ub, size=self.dim)
        curr_y = float(func(curr_x))
        self.eval_count += 1

        best_x = curr_x.copy()
        best_y = curr_y

        tabu_list = []
        max_tabu = max(10, self.dim)
        sigma = 0.15
        stagnation = 0
        neigh_size = min(10, max(4, self.dim))

        while self.eval_count < self.budget:
            candidates = []
            
            # Generate valid neighborhood candidates
            for _ in range(neigh_size * 2):
                if len(candidates) >= neigh_size:
                    break
                step = np.random.normal(0, 1, size=self.dim) * (sigma * domain_range)
                cand_x = np.clip(curr_x + step, lb, ub)

                # Check Tabu list
                is_tabu = False
                for tabu_center in tabu_list:
                    dist = np.linalg.norm(cand_x - tabu_center)
                    if dist < tabu_radius:
                        is_tabu = True
                        break
                
                if not is_tabu:
                    candidates.append(cand_x)

            if not candidates:
                # All candidates tabu: force random restart
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                sigma = 0.15
                stagnation = 0
                continue

            # Evaluate neighborhood
            best_neigh_x = None
            best_neigh_y = float("inf")

            for cand_x in candidates:
                if self.eval_count >= self.budget:
                    break
                y = float(func(cand_x))
                self.eval_count += 1

                if y < best_neigh_y:
                    best_neigh_y = y
                    best_neigh_x = cand_x.copy()

                if y < best_y:
                    best_y = y
                    best_x = cand_x.copy()

            if self.eval_count >= self.budget:
                break

            # Move to best neighbor
            if best_neigh_y < curr_y:
                curr_x = best_neigh_x.copy()
                curr_y = best_neigh_y
                sigma = min(1.2 * sigma, 0.4)
                stagnation = 0
            else:
                sigma *= 0.85
                stagnation += 1

            # Check for stagnation
            if stagnation > 25 or np.max(sigma) < 1e-5:
                # Add current local minimum to Tabu list
                if len(tabu_list) >= max_tabu:
                    tabu_list.pop(0)
                tabu_list.append(curr_x.copy())

                # Restart
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                sigma = 0.15
                stagnation = 0

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
