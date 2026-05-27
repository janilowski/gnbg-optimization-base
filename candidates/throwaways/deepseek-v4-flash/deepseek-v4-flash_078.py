# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1)-Evolution Strategy with step-size adaptation using the classic 1/5 success rule, tailored for black-box minimization.
# Search state: A single parent solution (best_x) and its objective value (best_y), current step size sigma (scalar, isotropic), a history of recent successes/failures (list of bools) to compute success rate over a sliding window.
# Candidate generation: Perturb parent with isotropic Gaussian noise scaled by sigma and the per-dimension bounds range (to make step size relative to domain width). The candidate is clamped to the search domain.
# Selection and replacement: The candidate replaces the parent if it yields a lower (better) objective value (elitist selection).
# Adaptation: After each evaluation, the success rate over a sliding window of the last N evaluations (N = 4 + floor(3*dim/10)) is computed. If success rate > 0.2, sigma is increased by a factor (1.1); if < 0.2, sigma is decreased by (0.9). This implements the 1/5 success rule to maintain a desired success probability.
# Exploration mechanisms: Random initialization uniformly in the domain. Isotropic Gaussian mutations provide exploration. Adaptation of sigma helps maintain exploration when needed (large sigma) and fine-tuning when near optimum (small sigma).
# Exploitation mechanisms: Elitist keeping of the best solution; small sigma values allow local exploitation. The step-size adaptation tends to decrease sigma when many failures occur (e.g., near optimum), promoting exploitation.
# Boundary handling: Candidate points are clamped (clipped) to the lower and upper bounds after generation, ensuring all evaluations are feasible.
# Budget strategy: The algorithm uses exactly the provided budget of evaluations. It stops generating new candidates when the number of evaluations reaches the budget. The initial random point counts as one evaluation.
# Closest known influences: Rechenberg's (1+1)-ES with 1/5 success rule. The implementation includes a sliding window for success rate rather than generational count.
# Novelty or unusual aspects: Simple, compact; uses dimension-dependent window size; scales step size by domain range to be more robust across different function scales.
# Failure modes: May converge prematurely on multimodal functions since it is a single-solution algorithm. The 1/5 rule may not be optimal for noisy functions. Clamping may cause loss of gradient information if optimum lies near boundary. Large sigma could overshoot and repeatedly clamp, causing slow progress.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
from collections import deque

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Function must provide lower and upper bounds via .lower/.upper or .bounds.lb/.bounds.ub")

        # Domain range for scaling step size
        domain_range = ub - lb

        # Initialize parent uniformly in bounds
        best_x = lb + np.random.uniform(0, 1, size=self.dim) * domain_range
        best_y = func(best_x)
        evaluations_used = 1

        # Parameters for step-size adaptation (1/5 rule)
        sigma = 0.2 * np.linalg.norm(domain_range) / np.sqrt(self.dim)  # isotropic step size
        window_size = max(5, int(4 + 0.3 * self.dim))
        success_history = deque(maxlen=window_size)

        # Main loop
        while evaluations_used < self.budget:
            # Generate candidate by isotropic Gaussian perturbation
            noise = np.random.normal(0, 1, size=self.dim)
            candidate = best_x + sigma * noise * domain_range  # scale by domain range
            # Boundary clamping
            candidate = np.clip(candidate, lb, ub)

            # Evaluate candidate
            cand_y = func(candidate)
            evaluations_used += 1

            # Success?
            if cand_y < best_y:
                best_x = candidate
                best_y = cand_y
                success_history.append(True)
            else:
                success_history.append(False)

            # Adapt sigma using 1/5 rule over sliding window
            if len(success_history) == window_size:
                success_rate = np.mean(success_history)
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                # If exactly 0.2, no change

        return best_x, best_y
