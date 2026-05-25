import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: (1+1)-Evolution Strategy with adaptive step size using the 1/5 success rule.
# Search state: current solution x (parent), overall best solution best_x, step size sigma,
#              and a deque of recent success flags for adaptation.
# Candidate generation: isotropic Gaussian mutation with step size sigma, then clamped to bounds.
# Selection and replacement: parent replaced only if child is strictly better (minimization);
#                            best overall updated accordingly.
# Adaptation: step size sigma adjusted every generation based on success rate over a fixed-length
#             history window (default 10). If rate > 0.2, sigma *= 1.2; if rate < 0.2, sigma /= 1.2.
# Exploration mechanisms: Gaussian mutation allows global search; adaptive sigma controls scale.
# Exploitation mechanisms: Elitist (1+1) selection focuses on best found, driving convergence.
# Boundary handling: each candidate is clipped to variable bounds.
# Budget strategy: initial evaluation consumes one; each subsequent candidate evaluation consumes one;
#                  loop stops when evaluations == budget.
# Closest known influences: Rechenberg's (1+1)-ES with 1/5 rule.
# Novelty or unusual aspects: None.
# Failure modes: May stagnate in rugged landscapes if step size shrinks prematurely; high-dimensional
#                problems may need a population-based method.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """A simple (1+1)-Evolution Strategy for black-box minimization."""

    def __init__(self, budget: int, dim: int):
        """
        Args:
            budget: maximum number of function evaluations.
            dim: dimensionality of the problem.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """Run the optimizer on a given function.

        Args:
            func: a callable that implements the GNBG test function interface.
                  Must expose lower/upper or bounds.lb/bounds.ub.

        Returns:
            (best_x, best_y) where best_x is a 1D numpy array and best_y is a float.
        """
        # ----- read bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        # ensure 1D arrays
        lb = np.atleast_1d(lb)
        ub = np.atleast_1d(ub)
        dim = self.dim

        # ----- initialisation -----
        x = np.random.uniform(lb, ub, size=dim)
        best_y = func(x)
        best_x = x.copy()
        evals = 1

        # step size: 20% of the range in each dimension (average)
        sigma = 0.2 * np.mean(ub - lb)
        # history for 1/5 rule
        history_len = min(10, max(5, self.budget // 20))
        success_history = []  # store booleans

        # ----- main loop -----
        while evals < self.budget:
            # candidate generation
            y = x + sigma * np.random.randn(dim)
            y = np.clip(y, lb, ub)          # boundary handling: clamp
            fy = func(y)
            evals += 1

            # update best
            if fy < best_y:
                best_y = fy
                best_x = y.copy()

            # success if better than parent (minimization)
            success = fy < func(x)   # parent value, no extra eval
            success_history.append(success)
            if len(success_history) > history_len:
                success_history.pop(0)

            # replace parent only if child is better (1+1-ES elitist)
            if success:
                x = y.copy()

            # --- step size adaptation (1/5 rule) ---
            if len(success_history) == history_len:
                success_rate = np.mean(success_history)
                if success_rate > 0.2:
                    sigma *= 1.2
                elif success_rate < 0.2:
                    sigma /= 1.2
                # optional: reset history to avoid jitter after adaptation
                success_history.clear()
                # tiny floor to prevent sigma becoming too small
                sigma = max(sigma, 1e-10)

            # safeguard: ensure we don't exceed budget inside loop
            if evals >= self.budget:
                break

        return best_x, best_y
