# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A pattern search (compass search) algorithm augmented with random diagonal probes to navigate continuous domains.
# Search state: Stores incumbent position, step size vector delta, best objective value seen, and stagnation metrics.
# Candidate generation: Proposes points along positive and negative unit coordinate vectors scaled by delta, plus randomized diagonal vectors.
# Selection and replacement: Replaces incumbent immediately upon discovering any strictly improving coordinate or diagonal step.
# Adaptation: Multiplies step size delta by 2 upon success; divides delta by 2 when all orthogonal search directions fail.
# Exploration mechanisms: Random restarts when delta drops below numerical precision threshold, and random diagonal probes during failed expansions.
# Exploitation mechanisms: Exact coordinate-wise alignment rapidly optimizes separable or axis-aligned landscape valleys.
# Boundary handling: All probed points are strictly clipped to the valid lower and upper domain boundaries.
# Budget strategy: Iterates through coordinate axes sequentially while maintaining strict evaluation budget caps.
# Closest known influences: Compass search / generalized pattern search (Torczon).
# Novelty or unusual aspects: Hybridizes exact orthogonal stencil evaluations with stochastic diagonal probing to overcome non-separable ridges.
# Failure modes: Scales linearly in cost with dimensionality per full stencil pass, slowing down convergence in very high dimensions.
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

        delta_scale = 0.2
        max_scale = 0.4
        min_scale = 1e-6

        while self.eval_count < self.budget:
            step_size = delta_scale * domain_range
            improved = False

            # Orthogonal search along each coordinate
            for i in range(self.dim):
                if self.eval_count >= self.budget or improved:
                    break

                for sign in [1, -1]:
                    if self.eval_count >= self.budget:
                        break

                    trial = curr_x.copy()
                    trial[i] = np.clip(trial[i] + sign * step_size[i], lb[i], ub[i])
                    y = float(func(trial))
                    self.eval_count += 1

                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

                    if y < curr_y:
                        curr_x = trial.copy()
                        curr_y = y
                        improved = True
                        break

            if improved:
                delta_scale = min(max_scale, delta_scale * 2.0)
            else:
                # Orthogonal search failed: try random diagonal probes
                for _ in range(min(5, self.dim)):
                    if self.eval_count >= self.budget or improved:
                        break
                    step = np.random.normal(0, 1, size=self.dim) * step_size
                    trial = np.clip(curr_x + step, lb, ub)
                    y = float(func(trial))
                    self.eval_count += 1

                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

                    if y < curr_y:
                        curr_x = trial.copy()
                        curr_y = y
                        improved = True
                        break

                if not improved:
                    delta_scale /= 2.0

            # Check for collapse
            if delta_scale < min_scale:
                curr_x = np.random.uniform(lb, ub, size=self.dim)
                if self.eval_count < self.budget:
                    curr_y = float(func(curr_x))
                    self.eval_count += 1
                    if curr_y < best_y:
                        best_y = curr_y
                        best_x = curr_x.copy()
                delta_scale = 0.2

        return best_x, best_y
