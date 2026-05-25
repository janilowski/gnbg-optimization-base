import numpy as np
from collections import deque

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A (1+1)-Evolution Strategy with self-adaptive step size and
#          reflection-based boundary handling for black-box minimization.
# Search state: Current best point (best_x) and a scalar step-size factor
#               (sigma) that scales per‑dimension ranges.
# Candidate generation: A trial point is produced by adding
#                       sigma * (upper-lower) * N(0,1) noise to best_x,
#                       then reflecting any out‑of‑bound coordinates
#                       back into the feasible domain.
# Selection and replacement: Deterministic – the candidate replaces the
#                            current best if and only if it yields a lower
#                            objective value.
# Exploration mechanisms: Initial random point; step‑size increases when
#                           the recent success rate exceeds 20%, allowing
#                           the search to jump out of local minima.
# Exploitation mechanisms: Step‑size decreases when success rate falls
#                           below 20%, enabling fine‑scale refinement near
#                           the current optimum.
# Boundary handling: Each violated coordinate is mirrored around the
#                    violated bound until it lies inside the domain.
# Budget strategy: One function evaluation per iteration (initial point
#                  plus each candidate budget exactly). The loop
#                  terminates when the evaluation count reaches the
#                  provided budget.
# Closest known influences: (1+1)-ES with Rechenberg's 1/5 rule.
# Novelty or unusual aspects: Uses a moving‑window success rate estimate
#                             instead of fixed‑length generations.
# Failure modes: On highly multimodal landscapes the single parent may
#                converge prematurely; very small budgets (<~10 evaluations)
#                may prevent any step‑size adaptation.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    """Minimisation of a black‑box function using a (1+1)-Evolution Strategy."""

    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimisation.

        Parameters
        ----------
        func : callable
            The objective function to minimise. It is assumed to expose
            either `.lower` / `.upper` or `.bounds.lb` / `.bounds.ub`.

        Returns
        -------
        best_x : np.ndarray
            Best point found.
        best_y : float
            Value of the objective at best_x.
        """
        # ------------------------------------------------------------------
        # 1. Retrieve domain bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.atleast_1d(np.asarray(func.lower, dtype=float))
            upper = np.atleast_1d(np.asarray(func.upper, dtype=float))
        elif hasattr(func, 'bounds'):
            # Some benchmarks store bounds as a Bounds object
            lower = np.atleast_1d(np.asarray(func.bounds.lb, dtype=float))
            upper = np.atleast_1d(np.asarray(func.bounds.ub, dtype=float))
        else:
            # Fallback (should never be triggered by GNBG)
            lower = np.full(self.dim, -1e10)
            upper = np.full(self.dim, 1e10)

        # Ensure shapes match the declared dimension
        if lower.ndim == 0:
            lower = np.full(self.dim, lower)
            upper = np.full(self.dim, upper)

        # ------------------------------------------------------------------
        # 2. Initialisation
        # ------------------------------------------------------------------
        best_x = lower + np.random.uniform(size=self.dim) * (upper - lower)
        best_y = func(best_x)
        evaluations = 1

        # ------------------------------------------------------------------
        # 3. Step‑size adaptation parameters
        # ------------------------------------------------------------------
        sigma = 0.2                     # relative step size (scaled by range)
        window_size = max(10, self.dim * 5)
        successes = deque(maxlen=window_size)

        # ------------------------------------------------------------------
        # 4. Main optimisation loop
        # ------------------------------------------------------------------
        while evaluations < self.budget:
            # --- candidate generation ---
            step = sigma * (upper - lower) * np.random.randn(self.dim)
            candidate = best_x + step

            # reflect out‑of‑bound coordinates
            for i in range(self.dim):
                while candidate[i] < lower[i] or candidate[i] > upper[i]:
                    if candidate[i] < lower[i]:
                        candidate[i] = lower[i] + (lower[i] - candidate[i])
                    if candidate[i] > upper[i]:
                        candidate[i] = upper[i] - (candidate[i] - upper[i])

            # --- evaluation and selection ---
            candidate_y = func(candidate)
            evaluations += 1

            if candidate_y < best_y:
                best_x, best_y = candidate, candidate_y
                successes.append(1)
            else:
                successes.append(0)

            # --- step‑size adaptation (window‑based 1/5 rule) ---
            if len(successes) == window_size:
                success_rate = sum(successes) / window_size
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9

        return best_x, best_y
