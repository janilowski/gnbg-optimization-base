import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a (1+1)-Evolution Strategy with the 1/5 success rule for continuous black-box minimization.
# Search state: A single best solution vector `best_x` and its objective value `best_y`. The step-size `sigma` is maintained globally.
# Candidate generation: Add isotropic Gaussian noise scaled by `sigma` to the current best vector, then clip to bounds.
# Selection and replacement: The candidate replaces the best if its objective value is strictly lower (minimization). No explicit archive or population.
# Adaptation: Every `adapt_freq` evaluations, the success rate over that window is computed. If > 0.2, sigma is increased (multiply by 1.21); otherwise decreased (multiply by 0.8). The frequency adapt_freq is set to min(10, budget//10 + 1) to work reasonably across budgets.
# Exploration mechanisms: Large initial sigma relative to the search space width (0.2 * range per dimension) and the automatic step-size adaptation encourage exploration when progress is rare.
# Exploitation mechanisms: When many recent steps succeed, sigma is increased (surprising but part of ES theory: high success rate indicates sigma too small? Actually classic rule: if success rate > 1/5, increase sigma to explore more; if < 1/5, decrease to focus. This balances exploitation and exploration).
# Boundary handling: Candidate coordinates are clipped to the lower and upper bounds.
# Budget strategy: The algorithm spends exactly `budget` evaluations (1 for initial, then budget-1 for iterations). No early stopping.
# Closest known influences: Classic (1+1) ES with 1/5 success rule (Rechenberg, 1973).
# Novelty or unusual aspects: No recombination or population. Relies entirely on step-size adaptation. Very simple and robust across dimensions if budget is reasonable.
# Failure modes: (1) Very low budget (< dim+1) may not allow proper adaptation, but the algorithm still returns the best of the few random samples. (2) Highly multimodal landscapes may cause premature convergence if sigma shrinks too fast. (3) If the optimum lies on the boundary, clipping may hinder progress, but that is common to many algorithms.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Minimization via (1+1)-Evolution Strategy with step-size adaptation."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # --- Read bounds ---------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower).ravel()
            ub = np.asarray(func.upper).ravel()
        else:
            lb = np.asarray(func.bounds.lb).ravel()
            ub = np.asarray(func.bounds.ub).ravel()

        # --- Parameter setup ------------------------------------------------
        budget = self.budget
        dim = self.dim
        # Initial step-size as 20% of the domain width
        domain_width = ub - lb
        sigma = 0.2 * np.min(domain_width)  # scalar step-size

        # Adaptation frequency: small windows for small budgets
        adapt_freq = max(2, min(10, budget // 10 + 1))
        success_counter = 0
        eval_count = 0

        # --- Initial point --------------------------------------------------
        best_x = lb + np.random.rand(dim) * domain_width
        best_y = func(best_x)
        eval_count = 1

        # --- Main loop ------------------------------------------------------
        # We need to spend exactly 'budget' evaluations. We have already spent 1.
        remaining = budget - 1
        while remaining > 0:
            # Generate candidate
            candidate = best_x + sigma * np.random.randn(dim)
            # Clip to bounds
            candidate = np.clip(candidate, lb, ub)

            # Evaluate
            candidate_y = func(candidate)
            eval_count += 1
            remaining -= 1

            # Selection
            if candidate_y < best_y:
                best_x = candidate
                best_y = candidate_y
                success_counter += 1

            # Step-size adaptation (1/5 rule)
            if eval_count % adapt_freq == 0:
                success_rate = success_counter / adapt_freq
                if success_rate > 0.2:
                    sigma *= 1.21
                else:
                    sigma *= 0.8
                # Keep sigma within reasonable bounds
                sigma = max(sigma, 1e-12 * np.min(domain_width))
                sigma = min(sigma, np.max(domain_width))
                success_counter = 0

        return best_x, best_y
