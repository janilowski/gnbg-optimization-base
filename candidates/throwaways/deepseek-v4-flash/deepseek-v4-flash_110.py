import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: (1+1)-Evolution Strategy with Rechenberg's 1/5 success rule for step size control. Designed for black-box minimization on the GNBG benchmark.
# Search state: Single candidate solution (best so far) and a step size sigma.
# Candidate generation: Add isotropic Gaussian noise scaled by sigma to the current best solution. (x_new = x_best + sigma * np.random.randn(dim))
# Selection and replacement: If the new candidate is better (lower objective value), replace the current best with it (greedy selection).
# Adaptation: Maintain a sliding window of the last L = max(1, dim) successes or failures. After each candidate evaluation, if the success rate over the window exceeds 0.2, sigma is multiplied by 1.2; if below 0.2, sigma is multiplied by 0.85. This implements Rechenberg's 1/5 rule.
# Exploration mechanisms: Gaussian mutations with step size adaptation.
# Exploitation mechanisms: Greedy selection (only accepts improvements), step size shrinking when success is rare focuses local search.
# Boundary handling: Projection (clip) to bounds.
# Budget strategy: Evaluate one candidate per iteration; loop until the initial evaluation plus generated candidates reach exactly the budget.
# Closest known influences: (1+1)-ES with Rechenberg's 1/5 rule.
# Novelty or unusual aspects: None. Standard textbook implementation.
# Failure modes: May get stuck in local optima due to greedy strategy; struggles with highly multimodal or rugged landscapes. Step size adaptation may be slow or unstable in very high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        """
        Args:
            budget (int): Maximum number of function evaluations.
            dim (int): Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimizer on the given objective function.

        Args:
            func: Callable that accepts a 1-D numpy array and returns a float.
                  Must expose bounds via either `func.lower` / `func.upper`
                  or `func.bounds.lb` / `func.bounds.ub`.

        Returns:
            tuple: (best_x, best_y) where best_x is a 1-D numpy array and
                   best_y is the corresponding objective value.
        """
        # ----- Read bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.array(func.lower, dtype=float)
            upper = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lower = np.array(b.lb, dtype=float)
            upper = np.array(b.ub, dtype=float)
        else:
            raise AttributeError("func must have lower/upper or bounds.lb/ub")

        dim = self.dim
        # ----- Initialisation -----
        x = lower + np.random.rand(dim) * (upper - lower)          # random start
        y = func(x)
        best_x = x.copy()
        best_y = y
        remaining = self.budget - 1                                # one evaluation used

        # ----- Algorithm parameters -----
        range_span = np.median(upper - lower)                      # typical scale
        sigma = 0.2 * range_span                                   # initial step size
        window_len = max(1, dim)                                   # history for 1/5 rule
        success_hist = []                                          # sliding window of bools

        while remaining > 0:
            # Generate candidate
            candidate = best_x + sigma * np.random.randn(dim)
            candidate = np.clip(candidate, lower, upper)           # boundary handling

            # Evaluate (never exceed budget)
            y_candidate = func(candidate)
            remaining -= 1

            # Selection
            improved = y_candidate < best_y
            if improved:
                best_x = candidate.copy()
                best_y = y_candidate

            # Update success history
            success_hist.append(improved)
            if len(success_hist) > window_len:
                success_hist.pop(0)

            # Apply Rechenberg's 1/5 rule when we have enough history
            if len(success_hist) >= window_len:
                success_rate = sum(success_hist) / window_len
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma *= 0.85
                # else keep sigma unchanged

            # Prevent sigma from collapsing completely
            sigma = max(sigma, 1e-8 * range_span)

        return best_x, best_y
