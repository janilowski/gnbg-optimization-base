# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a minimalist (1+1) Evolution Strategy with step‑size adaptation (the 1/5 rule) and a stagnation‑triggered restart to balance exploration and exploitation on black‑box minimization problems.
# Search state: Current best solution vector (x), mutation step size (sigma), sliding‑window success counter, and a stagnation counter.
# Candidate generation: A candidate is produced by adding Gaussian noise (σ·N(0,I)) to the current solution.
# Selection and replacement: The candidate is accepted only if it yields a lower objective value than the current best; otherwise the current solution is retained.
# Adaptation: Every few evaluations (a configurable window) the success rate is evaluated; sigma is increased by a factor 1.1 when the success rate exceeds 1/5, otherwise decreased by 0.9, keeping sigma within safe bounds.
# Exploration mechanisms: Large sigma encourages broad exploration; periodic restarts with fresh random points kick the search out of possible local optima.
# Exploitation mechanisms: Small sigma focuses refinement around the best‑found point.
# Boundary handling: After mutation each component is clipped to the problem’s lower/upper bounds.
# Budget strategy: Evaluations are counted explicitly; the algorithm terminates as soon as the supplied budget is exhausted.
# Closest known influences: Classic (1+1)-ES with Rechenberg’s 1/5 rule; similar ideas appear in simple CMA‑ES but much simplified.
# Novelty or unusual aspects: A stagnation counter triggers a full restart (new random point and sigma reset), which is not part of the textbook (1+1)-ES but helps on multi‑modal landscapes.
# Failure modes: With very limited budgets the optimizer may not converge; sigma may become too large or too small in high dimensions, potentially degrading performance.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple (1+1)-ES with 1/5 rule adaptation and restart on stagnation.
    Suitable for low‑budget black‑box minimization tasks.
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def _random_point(self, lower, upper):
        """Return a uniform random point inside the bounds."""
        return lower + np.random.random(self.dim) * (upper - lower)

    def __call__(self, func):
        """
        Run the optimizer on the provided objective function.

        Parameters
        ----------
        func : callable
            Black‑box function that accepts a 1‑D numpy array and returns a scalar.
            Must expose either ``lower``/``upper`` or ``bounds.lb``/``bounds.ub``.

        Returns
        -------
        best_x : numpy.ndarray
            Best solution found.
        best_y : float
            Corresponding objective value.
        """
        # ----- Determine problem bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("func must have 'lower'/'upper' or 'bounds' attribute.")

        # ----- Initial random solution -----
        x = self._random_point(lower, upper)
        y = func(x)
        best_x = x.copy()
        best_y = float(y)
        evals = 1

        # ----- Mutation step size (initial guess) -----
        sigma = np.mean(upper - lower) * 0.5

        # ----- Adaptation parameters -----
        window = max(10, 2 * self.dim)          # how many attempts between sigma updates
        successes = 0
        attempts = 0

        # ----- Stagnation handling -----
        stagnation_limit = 5 * self.dim        # restart after this many failed attempts
        no_improve = 0

        # ----- Main optimization loop -----
        while evals < self.budget:
            # Generate candidate by Gaussian mutation
            candidate = x + sigma * np.random.randn(self.dim)
            # Enforce bounds
            candidate = np.clip(candidate, lower, upper)

            # Evaluate candidate
            cand_y = func(candidate)
            evals += 1

            # ----- Selection -----
            if cand_y < best_y:
                best_x = candidate.copy()
                best_y = float(cand_y)
                x = candidate
                successes += 1
                no_improve = 0
            else:
                no_improve += 1

            attempts += 1

            # ----- 1/5 rule adaptation of sigma -----
            if attempts % window == 0:
                if successes / window > 0.2:
                    sigma *= 1.1      # increase step size when success rate is high
                else:
                    sigma *= 0.9      # decrease step size when success rate is low
                successes = 0

            # Keep sigma in a reasonable range (avoid too small/large steps)
            sigma = np.clip(sigma, 1e-12, np.mean(upper - lower) * 10)

            # ----- Restart on stagnation -----
            if no_improve >= stagnation_limit:
                # Ensure we still have budget for a fresh evaluation
                if evals >= self.budget:
                    break
                # Draw a new random starting point
                x = self._random_point(lower, upper)
                y = func(x)
                evals += 1

                # Update best if this new point is better
                if y < best_y:
                    best_x = x.copy()
                    best_y = float(y)

                # Reset sigma and counters
                sigma = np.mean(upper - lower) * 0.5
                no_improve = 0
                successes = 0
                attempts = 0

        return best_x, best_y
