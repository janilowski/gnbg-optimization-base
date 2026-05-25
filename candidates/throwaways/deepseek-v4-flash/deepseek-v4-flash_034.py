import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1) Evolution Strategy (ES) with step-size adaptation using
#          Rechenberg's 1/5 rule. The algorithm maintains a single parent
#          solution and generates one offspring per iteration via isotropic
#          Gaussian mutation. Step size is updated periodically based on the
#          success rate over a fixed window. Bounds are enforced by reflection,
#          and the evaluation budget is strictly respected.
# Search state: A single best solution vector (parent_x) and its objective
#               value (parent_y), a step-size scalar (sigma), and a window of
#               recent success/failure flags for step-size adaptation.
# Candidate generation: Offspring = parent_x + sigma * N(0, I). Each
#                       component is adjusted for bounds via reflection.
# Selection and replacement: If the offspring yields a lower objective value
#                            (minimization), it replaces the parent.
# Adaptation: Every lambda_g (generation) the success rate over the last
#             lambda_g steps is computed. If the rate > 1/5, sigma is increased
#             by a factor; if < 1/5, sigma is decreased. This is the classic
#             1/5 rule.
# Exploration mechanisms: Isotropic Gaussian mutation with dynamic step size.
#                         Large sigma early, smaller later if convergence.
# Exploitation mechanisms: The step size shrinks when success rate is low
#                          (indicating many failures) and grows when many
#                          successes occur, balancing exploration/exploitation.
# Boundary handling: Reflection – if a coordinate of the offspring falls
#                    outside [lower, upper], it is reflected back into the
#                    feasible domain (mirroring). If the reflected point
#                    still falls outside (e.g., due to very large step), it
#                    is randomly re-sampled within the bounds.
# Budget strategy: The algorithm stops when all evaluations (initial + every
#                  offspring) reach budget-1 (the initial counts as one).
#                  It does not exceed budget.
# Closest known influences: Rechenberg's (1+1) Evolution Strategy, Schwefel's
#                           step-size adaptation.
# Novelty or unusual aspects: A very simple, compact implementation with
#                             reflection for bounds and a window-based success
#                             rate that resets to avoid stale statistics.
# Failure modes: May converge prematurely on highly multimodal landscapes
#                due to the lack of population diversity. Step-size adaptation
#                can oscillate if budget is too short to stabilize.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Minimization benchmark algorithm: (1+1)-ES with Rechenberg's rule."""

    def __init__(self, budget: int, dim: int):
        """
        Args:
            budget: Maximum number of objective function evaluations allowed.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # Fixed hyperparameters for the (1+1)-ES
        self.lambda_g = 10              # Window size for success rate
        self.sigma_init_factor = 0.3    # Initial sigma as fraction of the median range
        self.sigma_min = 1e-8           # Minimum step size to avoid numerical issues
        self.sigma_max = 1e3            # Maximum step size
        self.inc_factor = 1.2           # Multiply sigma when success rate > 1/5
        self.dec_factor = 0.82          # Multiply sigma when success rate < 1/5

    def __call__(self, func):
        """
        Run the optimization.

        Args:
            func: A callable object that returns a scalar objective value.
                  It may have attributes .lower / .upper or .bounds.lb / bounds.ub
                  to define the feasible domain.

        Returns:
            (best_x, best_y): The best found solution and its objective.
        """
        # ---------- extract bounds ----------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # fallback – assume hypercube [-5,5]^d
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim,  5.0)

        # Ensure bounds are vectors of correct dimension
        if lb.ndim == 0:
            lb = np.repeat(lb, self.dim)
        if ub.ndim == 0:
            ub = np.repeat(ub, self.dim)

        # Median range for step size initialisation
        diff = ub - lb
        diff = np.where(diff <= 0, 1.0, diff)         # safeguard constant dimensions
        range_median = np.median(diff)

        # ---------- Initialise ----------
        parent_x = lb + np.random.uniform(0, 1, size=self.dim) * diff
        parent_y = func(parent_x)                     # first evaluation
        evals_done = 1

        sigma = self.sigma_init_factor * range_median
        # Clamp sigma to [sigma_min, sigma_max]
        sigma = min(max(sigma, self.sigma_min), self.sigma_max)

        # Success rate window (circular buffer)
        success_window = np.zeros(self.lambda_g, dtype=bool)
        window_pos = 0
        window_filled = False

        # ---------- Main loop ----------
        while evals_done < self.budget:
            # Generate offspring
            step = np.random.randn(self.dim) * sigma
            offspring_x = parent_x + step

            # Boundary handling – reflection
            for i in range(self.dim):
                while True:
                    if offspring_x[i] < lb[i] or offspring_x[i] > ub[i]:
                        # reflect around the nearer boundary
                        if offspring_x[i] < lb[i]:
                            offspring_x[i] = 2 * lb[i] - offspring_x[i]
                        else:
                            offspring_x[i] = 2 * ub[i] - offspring_x[i]
                        # If still out of bounds (e.g., huge step), random reset
                        if offspring_x[i] < lb[i] or offspring_x[i] > ub[i]:
                            offspring_x[i] = lb[i] + np.random.uniform(0, 1) * diff[i]
                    else:
                        break

            # Evaluate offspring
            offspring_y = func(offspring_x)
            evals_done += 1

            # Selection
            improved = offspring_y < parent_y
            if improved:
                parent_x = offspring_x
                parent_y = offspring_y

            # Record success/failure in the window
            success_window[window_pos] = improved
            window_pos = (window_pos + 1) % self.lambda_g
            if window_pos == 0:
                window_filled = True

            # Step-size adaptation (Rechenberg's 1/5 rule) every lambda_g steps
            if window_filled and window_pos == 0:
                # Window is full and we just finished a full cycle
                success_rate = np.mean(success_window)
                if success_rate > 0.2:
                    sigma *= self.inc_factor
                elif success_rate < 0.2:
                    sigma *= self.dec_factor
                # Clamp sigma
                sigma = min(max(sigma, self.sigma_min), self.sigma_max)

            # Optional emergency: if sigma is extremely small and no improvement
            # for a long time, try a restart – skip for simplicity.

        # Return the best found solution
        return (parent_x, parent_y)
