import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a (1+1)-Evolution Strategy with step-size adaptation via Rechenberg's 1/5 success rule.
# Search state: A single candidate solution (x) and its objective value (y), plus the best seen so far.
# Candidate generation: Adds isotropic Gaussian noise scaled by a global step size sigma.
# Selection and replacement: Greedy – the offspring replaces the parent if it yields a lower objective value (minimization).
# Adaptation: Every 5 generations, sigma is increased by 10% if the empirical success rate exceeds 0.2, decreased by 10% otherwise.
# Exploration mechanisms: Gaussian mutation with dynamic sigma; large sigma encourages exploration.
# Exploitation mechanisms: Greedy selection and tightening of sigma when success rate is low.
# Boundary handling: Candidate solutions are clipped to the variable bounds.
# Budget strategy: A single while loop enforces that the total number of function evaluations never exceeds the given budget.
# Closest known influences: Classic (1+1)-ES with Rechenberg's rule.
# Novelty or unusual aspects: None – this is a textbook baseline.
# Failure modes: May converge prematurely on multi-modal landscapes; step-size adaptation can stagnate if sigma becomes too small; no restart mechanism.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """A (1+1)-Evolution Strategy for black-box minimization with step-size adaptation."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget  # maximum number of function evaluations
        self.dim = dim        # dimensionality of the search space
        # algorithmic parameters
        self.initial_sigma_rel = 0.1          # relative initial step size
        self.success_rate_target = 0.2        # desired success probability
        self.generation_window = 5            # adaptation frequency
        # state (reset in __call__)
        self.sigma = None
        self.success_counter = 0
        self.generation_counter = 0

    def __call__(self, func):
        # ----- determine bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            # safe fallback (should not happen in GNBG benchmark)
            lb = np.full(self.dim, -1e6)
            ub = np.full(self.dim, 1e6)

        # ----- initialisation -----
        range_size = ub - lb
        # avoid division by zero for degenerate dimensions
        range_size[range_size == 0] = 1.0

        # scale step size proportionally to the average range
        self.sigma = self.initial_sigma_rel * np.mean(range_size)

        # first candidate: uniform random in the domain
        x = lb + np.random.uniform(0, 1, size=self.dim) * range_size
        y = func(x)                     # evaluation
        best_x = x.copy()
        best_y = y
        evals = 1

        # reset counters
        self.success_counter = 0
        self.generation_counter = 0

        # ----- optimisation loop -----
        while evals < self.budget:
            # generate offspring via Gaussian mutation
            offspring = x + np.random.normal(0, self.sigma, size=self.dim)
            offspring = np.clip(offspring, lb, ub)   # boundary handling
            y_off = func(offspring)
            evals += 1

            # (1+1) selection
            if y_off < y:
                x = offspring
                y = y_off
                self.success_counter += 1
                if y_off < best_y:
                    best_x = offspring
                    best_y = y_off
            self.generation_counter += 1

            # step-size adaptation (every generation_window generations)
            if self.generation_counter >= self.generation_window:
                success_rate = self.success_counter / self.generation_counter
                if success_rate > self.success_rate_target:
                    self.sigma *= 1.1   # increase step size
                elif success_rate < self.success_rate_target:
                    self.sigma *= 0.9   # decrease step size
                # reset counters
                self.success_counter = 0
                self.generation_counter = 0
                # guard against numerical underflow
                if self.sigma < 1e-15:
                    self.sigma = 1e-15

        return best_x, best_y
