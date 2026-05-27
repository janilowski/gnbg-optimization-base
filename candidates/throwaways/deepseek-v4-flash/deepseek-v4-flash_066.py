# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a (1+1)-Evolution Strategy with adaptive step size and restarts for black-box minimization.
# Search state: Single candidate point (best_x) and its objective value (best_y). Also step size sigma (isotropic) and a success history buffer.
# Candidate generation: Add Gaussian noise scaled by sigma to the current best point, then clip to bounds.
# Selection and replacement: If new point yields lower objective value (minimization), replace best_x and best_y.
# Adaptation: Step size sigma is adapted using the (1/5)-th success rule: track success rate over a sliding window. If success rate > 0.2, increase sigma (multiply by 1.2); if < 0.2, decrease sigma (multiply by 0.85). Adjustments are damped.
# Exploration mechanisms: Initial random sampling; large sigma early; occasional restarts when sigma becomes negligible.
# Exploitation mechanisms: Local refinement via small sigma and selection of best point.
# Boundary handling: Candidate is clipped component-wise to [lower, upper] bounds. If all components equal to bounds after clipping, the candidate is still accepted (might be boundary optimum).
# Budget strategy: Evaluate no more than budget function calls; stop when budget exhausted.
# Closest known influences: Classic (1+1)-ES with Rechenberg's 1/5 rule; restart strategy from CMA-ES.
# Novelty or unusual aspects: Uses a success window rather than cumulative path length; restarts when step size drops too low to recover.
# Failure modes: May stall if the optimum lies on the boundary and the step size becomes too small; may require more budget for high-dimensional problems; fixed adaptation parameters may not suit all landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ---- read bounds from func ----
        try:
            lower = np.broadcast_to(func.lower, self.dim)
            upper = np.broadcast_to(func.upper, self.dim)
        except AttributeError:
            lower = np.broadcast_to(func.bounds.lb, self.dim)
            upper = np.broadcast_to(func.bounds.ub, self.dim)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        rng = np.random.default_rng()

        # ---- initialisation ----
        best_x = lower + rng.random(self.dim) * (upper - lower)
        best_y = func(best_x)
        evals = 1

        # step size (isotropic scalar)
        sigma = 0.2 * (upper - lower).mean()

        # sliding window for success rate
        window_size = max(10, min(100, self.budget // 10))
        successes = []                     # boolean list of recent updates

        # adaptation factors
        factor_up = 1.2
        factor_down = 0.85

        while evals < self.budget:
            # ---- generate candidate ----
            candidate = best_x + sigma * rng.normal(size=self.dim)
            candidate = np.clip(candidate, lower, upper)   # enforce bounds
            y_candidate = func(candidate)
            evals += 1

            # ---- selection ----
            success = y_candidate < best_y
            if success:
                best_x, best_y = candidate, y_candidate

            # ---- success tracking ----
            successes.append(success)
            if len(successes) > window_size:
                successes.pop(0)

            # ---- step size adaptation (1/5 rule) ----
            if len(successes) == window_size:
                success_rate = sum(successes) / window_size
                if success_rate > 0.2:
                    sigma *= factor_up
                elif success_rate < 0.2:
                    sigma *= factor_down

            # ---- restart if sigma becomes too small ----
            if sigma < 1e-12 * (upper - lower).mean():
                best_x = lower + rng.random(self.dim) * (upper - lower)
                best_y = func(best_x)
                evals += 1
                sigma = 0.2 * (upper - lower).mean()
                successes.clear()

        return best_x, best_y
