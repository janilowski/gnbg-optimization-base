# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: (1+1)-Evolution Strategy with step-size adaptation using Rechenberg’s 1/5 rule.
# Search state: Best solution (stored in normalized coordinates [0,1]^dim), a scalar step size in the normalized space, and a counter tracking recent successes.
# Candidate generation: Add isotropic Gaussian noise scaled by the step size to the current best normalized solution.
# Selection and replacement: Deterministic acceptance if the candidate yields a lower function value (minimization).
# Adaptation: Every dim evaluations, compute the success rate over the last dim evaluations; if >0.2 increase step size by factor 1.2, if <0.2 decrease by factor 0.85.
# Exploration mechanisms: Large step sizes produce distant candidates, encouraging global search.
# Exploitation mechanisms: Small step sizes refine the current best solution when few improvements occur.
# Boundary handling: Candidate normalized coordinates are clamped to [0,1] before mapping to the real domain.
# Budget strategy: One evaluation per candidate; the loop terminates when the evaluation budget is exhausted.
# Closest known influences: Rechenberg's (1+1)-ES, classic evolutionary strategies.
# Novelty or unusual aspects: None.
# Failure modes: May stagnate in rugged landscapes or high dimensions; step size can shrink to zero; no population diversity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Minimize a black‑box function using a (1+1)-Evolution Strategy."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from the function object
        try:
            lower = np.array(func.lower, dtype=float)
            upper = np.array(func.upper, dtype=float)
        except AttributeError:
            try:
                lower = np.array(func.bounds.lb, dtype=float)
                upper = np.array(func.bounds.ub, dtype=float)
            except AttributeError:
                raise AttributeError("Cannot read bounds from func.")

        dim = self.dim
        # Work in normalized space [0,1]^dim for easier step‑size control
        range_ = upper - lower

        # Initial solution
        best_norm = np.random.uniform(0.0, 1.0, dim)
        best_real = lower + best_norm * range_
        best_y = func(best_real)
        evals = 1

        # Step size in normalized space, initial ~20% of the box size
        step_size = 0.2
        min_step = 1e-8
        success_cnt = 0

        # Main loop: one candidate per evaluation until budget exhausted
        while evals < self.budget:
            # Generate candidate via isotropic Gaussian perturbation
            candidate_norm = best_norm + step_size * np.random.randn(dim)
            # Clamp to [0,1]
            candidate_norm = np.clip(candidate_norm, 0.0, 1.0)
            candidate_real = lower + candidate_norm * range_
            candidate_y = func(candidate_real)
            evals += 1

            # Selection
            if candidate_y < best_y:
                best_norm = candidate_norm
                best_real = candidate_real
                best_y = candidate_y
                success_cnt += 1

            # Adapt step size every dim evaluations
            if evals % dim == 0:
                success_rate = success_cnt / dim
                if success_rate > 0.2:
                    step_size *= 1.2
                elif success_rate < 0.2:
                    step_size *= 0.85
                step_size = max(step_size, min_step)
                success_cnt = 0

        return best_real, best_y
