import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact (1+1)-Evolutionary Strategy with step-size adaptation and random restarts.
#          Designed for black-box minimization on any budget and dimension up to moderate size.
# Search state: A single current solution (parent) and its objective value, plus a step-size sigma
#               (per dimension isotropic). A history of recent improvement success flags is maintained
#               to adapt sigma via the classical 1/5 success rule. A counter tracks stalling to trigger restarts.
# Candidate generation: One offspring per iteration: parent + sigma * Gaussian(0,I) clipped to bounds.
# Selection and replacement: Greedy – offspring replaces parent if it yields a strictly better objective.
# Adaptation: Every 50 evaluations, the success fraction is computed. If above 0.2, sigma is multiplied by 1.2;
#             if below 0.2, sigma is multiplied by 0.8. Sigma is clamped to a minimal positive value to avoid collapse.
# Exploration mechanisms: Large initial sigma (~20% of the range) and dynamic step-size adjustments.
#                         Random restarts reinitialize the parent uniformly in the search space,
#                         providing global exploration.
# Exploitation mechanisms: Small sigma values after many failures focus on local refinement around the best
#                          solution found so far. Greedy selection ensures monotonic improvement along each restart.
# Boundary handling: Any coordinate that falls outside [lower, upper] is clipped to the nearest bound.
# Budget strategy: Total function evaluations are tracked. The algorithm stops immediately when the budget is
#                  exhausted, even mid-iteration. Restarts are scheduled based on stall counter (no improvement
#                  for 100 evaluations), but only if budget remains.
# Closest known influences: Rechenberg’s (1+1)-ES with 1/5-rule; random restarts as a common globalisation method.
# Novelty or unusual aspects: Very compact implementation, no population, no crossover. Relies entirely on
#                             step-size control and multiple restarts to avoid local optima.
# Failure modes: On highly multi-modal landscapes with strong local optima, restarts may not find the global
#                optimum within the budget. Since only one candidate per iteration is used, progress slows in
#                high dimensions because isotropic Gaussian steps cannot align with steepest gradients.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """A simple (1+1)-Evolutionary Strategy with step-size adaptation and restarts.

    Parameters
    ----------
    budget : int
        Maximum number of function evaluations.
    dim : int
        Dimensionality of the search space.

    The algorithm maintains a single parent, applies isotropic Gaussian mutation,
    uses the 1/5 success rule to adapt step size, and restarts from a random
    position when no improvement is seen for a fixed number of evaluations.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ------------------------------------------------------------
        # 1. Extract bounds (support both naming conventions)
        # ------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective function must provide .lower/.upper or .bounds.lb/.bounds.ub")

        # ------------------------------------------------------------
        # 2. Algorithm parameters
        # ------------------------------------------------------------
        dim = self.dim
        budget = self.budget
        # Restart threshold: if no improvement for this many evaluations, restart
        restart_stall = max(100, 10 * dim)
        # Adaptation window size (number of past evaluations used to compute success rate)
        adapt_window = max(50, dim * 5)
        # Safety epsilon for sigma minimum
        sigma_min = 1e-12

        # Helper for random initialization
        def random_solution():
            return lb + np.random.rand(dim) * (ub - lb)

        # Counters
        evals = 0
        evaluations_since_last_improvement = 0

        # ------------------------------------------------------------
        # 3. Initialise first restart
        # ------------------------------------------------------------
        parent = random_solution()
        parent_y = float(func(parent))
        evals += 1

        best_x = parent.copy()
        best_y = parent_y

        # History of successes (1 = improvement, 0 = no improvement)
        success_history = np.zeros(adapt_window, dtype=bool)
        # Step size – 20% of the average range
        sigma = float(np.mean(ub - lb)) * 0.2

        # ------------------------------------------------------------
        # 4. Main loop
        # ------------------------------------------------------------
        while evals < budget:
            # --- Check for restart ---
            if evaluations_since_last_improvement >= restart_stall:
                # Restart: reset parent, step size, history, and stall counter
                parent = random_solution()
                parent_y = float(func(parent))
                evals += 1
                if evals > budget:
                    break
                if parent_y < best_y:
                    best_x = parent.copy()
                    best_y = parent_y
                sigma = float(np.mean(ub - lb)) * 0.2
                success_history[:] = False
                evaluations_since_last_improvement = 0
                continue

            # --- Generate offspring ---
            # Gaussian mutation, isotropic
            child = parent + sigma * np.random.randn(dim)
            # Clip to bounds
            child = np.clip(child, lb, ub)
            child_y = float(func(child))
            evals += 1

            # --- Selection and update ---
            improvement = child_y < parent_y
            if improvement:
                # Replace parent
                parent = child
                parent_y = child_y
                if parent_y < best_y:
                    best_x = parent.copy()
                    best_y = parent_y
                evaluations_since_last_improvement = 0
            else:
                evaluations_since_last_improvement += 1

            # --- Update success history (circular buffer) ---
            success_history = np.roll(success_history, -1)
            success_history[-1] = improvement

            # --- Adapt step size every `adapt_window` evaluations ---
            if evals % adapt_window == 0:
                success_rate = np.mean(success_history)
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.8
                # Clamp sigma to avoid numerical issues
                sigma = max(sigma, sigma_min)

        return best_x, best_y
