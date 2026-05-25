import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1) Evolution Strategy (ES) with Rechenberg's 1/5 success rule
# for step-size adaptation, designed for black-box minimization on the GNBG benchmark.
# Search state: A single best candidate solution (best_x) and its objective value (best_y),
# plus a global step size sigma and a counter tracking successful mutations over a fixed window.
# Candidate generation: Best solution is mutated by adding isotropic Gaussian noise scaled by sigma.
# Selection and replacement: Greedy – the new candidate replaces the best if it yields a lower
# (or equal) objective value (minimization).
# Adaptation: Every N generations (window = 10), the success rate (fraction of accepted mutations)
# is compared to 1/5. If above, sigma is increased (×1.2); if below, sigma is decreased (×0.8).
# Exploration mechanisms: large sigma early in the run encourages broad exploration; the
# multiplicative adaptation can recover after convergence.
# Exploitation mechanisms: As success rate drops near an optimum, sigma shrinks to allow
# finer local tuning.
# Boundary handling: New candidate coordinates are clamped to the lower and upper bounds.
# Budget strategy: One initial evaluation, then exactly budget-1 additional evaluations in
# the main loop; no function evaluations are wasted.
# Closest known influences: Rechenberg's (1+1)-ES, Schwefel's mutation step control.
# Novelty or unusual aspects: None – implementation follows the classic textbook algorithm.
# Failure modes: Can stagnate in rugged landscapes or plateaus; isotropic mutation may be
# inefficient for ill-conditioned problems; greedy selection can cause premature convergence
# in the presence of noise (no resampling used).
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """(1+1) Evolution Strategy with Rechenberg's 1/5 rule for step-size adaptation."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ---------- read bounds ----------
        try:
            lower = np.array(func.lower, dtype=float)
            upper = np.array(func.upper, dtype=float)
        except AttributeError:
            try:
                lower = np.array(func.bounds.lb, dtype=float)
                upper = np.array(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Cannot find domain bounds from func")

        dim = self.dim
        # ---------- parameter settings ----------
        range_ = upper - lower
        # initial step size: 20% of the average domain range
        sigma = 0.2 * np.mean(range_)
        # Rechenberg's window length
        window = 10
        success_cnt = 0
        gen = 0

        # ---------- initialisation ----------
        best_x = np.random.uniform(lower, upper, size=dim)
        best_y = func(best_x)
        evaluations = 1

        # ---------- main evolution loop ----------
        while evaluations < self.budget:
            # generate candidate by isotropic Gaussian mutation
            candidate = best_x + sigma * np.random.randn(dim)
            # project back into bounds by clamping
            candidate = np.clip(candidate, lower, upper)

            cand_y = func(candidate)
            evaluations += 1

            # greedy selection
            if cand_y <= best_y:
                best_x = candidate
                best_y = cand_y
                success_cnt += 1

            # step-size adaptation every 'window' generations
            gen += 1
            if gen % window == 0:
                success_rate = success_cnt / window
                if success_rate > 0.2:
                    sigma *= 1.2   # increase step
                else:
                    sigma *= 0.8   # decrease step
                # reset success counter
                success_cnt = 0

        return best_x, best_y
