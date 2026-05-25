import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple (1+1) Evolution Strategy with adaptive step size for black-box minimization.
# Search state: Current point x, its fitness f(x), best overall solution (x_best, f_best), and a scalar step size sigma.
# Candidate generation: Offspring = x + sigma * N(0, I), then clipped to the feasible box.
# Selection and replacement: Offspring replaces the current point if it has lower (better) objective value.
# Adaptation: Step size is increased by a factor of 1.2 after a successful mutation and decreased by 0.8 after a failure, with a lower bound of 1e-10.
# Exploration mechanisms: Gaussian mutations with an adaptive step size that grows after successes, allowing larger jumps.
# Exploitation mechanisms: Step size shrinks after repeated failures, focusing local search around the current point.
# Boundary handling: Offspring coordinates are clipped to [lower, upper] bounds.
# Budget strategy: The initial point evaluation consumes one budget unit; each offspring evaluation consumes one unit. The loop continues until budget exhaustion.
# Closest known influences: (1+1)-ES with the 1/5th success rule; simple adaptive Evolution Strategy.
# Novelty or unusual aspects: None; the implementation is deliberately minimal and uses only the standard adaptation heuristic.
# Failure modes: Premature convergence to suboptimal solutions due to the lack of population diversity. The scalar step size may not adapt well to different coordinate scales, leading to slow progress in high‑dimensional or ill‑conditioned problems.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ---------- bounds ----------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lower = np.asarray(b.lb, dtype=float)
            upper = np.asarray(b.ub, dtype=float)
        else:
            raise AttributeError("Cannot determine bounds from func")
        # ensure arrays of correct shape
        lower = lower.ravel()
        upper = upper.ravel()
        if lower.size != self.dim:
            raise ValueError(f"lower bound size {lower.size} != dim {self.dim}")
        if upper.size != self.dim:
            raise ValueError(f"upper bound size {upper.size} != dim {self.dim}")

        # ---------- initialisation ----------
        rng = np.random.default_rng()  # harness seeds before call
        x = rng.uniform(lower, upper, size=self.dim)
        fx = func(x)
        best_x = x.copy()
        best_y = fx
        remaining = self.budget - 1

        # initial step size (20% of the average range)
        sigma = 0.2 * (upper - lower).mean()

        # ---------- main loop ----------
        while remaining > 0:
            z = rng.normal(0, 1, size=self.dim)
            trial = x + sigma * z
            trial = np.clip(trial, lower, upper)
            ftrial = func(trial)
            remaining -= 1

            # success?
            if ftrial < fx:
                x = trial
                fx = ftrial
                sigma *= 1.2  # increase step size
            else:
                sigma *= 0.8  # decrease step size

            sigma = max(sigma, 1e-10)  # avoid zero step

            if ftrial < best_y:
                best_y = ftrial
                best_x = trial.copy()

        return best_x, best_y
