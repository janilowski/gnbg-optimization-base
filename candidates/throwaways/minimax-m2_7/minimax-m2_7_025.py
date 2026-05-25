# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a simple two‑phase search algorithm. First, it performs a uniform random sampling across the problem bounds to obtain a reasonable initial guess. Then it runs a hill‑climbing local search that perturbs the current solution with Gaussian steps, adapting the step size based on success/failure streaks. If progress stalls (step becomes negligible), the algorithm re‑initializes from a new random point.
# Search state: The algorithm keeps track of the overall best solution found (best_x, best_y) and the current point used for local exploration (x, y). It also maintains a mutable step size and a counter of consecutive failures.
# Candidate generation: Random points for the initial sampling; Gaussian perturbations scaled by the current step for hill climbing.
# Selection and replacement: A candidate is accepted as the new current point if it improves the current objective value. The best solution is updated whenever a better candidate is found.
# Adaptation: The step size grows by a factor of 1.1 after a successful move (up to a maximum of half the bound range) and shrinks by a factor of 0.5 after five consecutive failures. If the step falls below a tiny fraction of the bounds range, a restart is triggered.
# Exploration mechanisms: Broad uniform sampling at startup and random restarts when the local search stalls.
# Exploitation mechanisms: Fine‑grained Gaussian moves with adaptive scaling to concentrate effort near promising regions.
# Boundary handling: All generated points are clipped to the problem’s lower and upper bounds.
# Budget strategy: The budget is split: a small fraction (≈20% or dim*5 points, whichever is smaller) is used for initial sampling, and the remaining evaluations drive the hill‑climbing phase. The algorithm never exceeds the budget.
# Closest known influences: Classic (1+1)-evolutionary strategy with a 1/5 rule and simple restart heuristic.
# Novelty or unusual aspects: Combining uniform sampling with an adaptive Gaussian local search and using a restart triggered by step‑size decay.
# Failure modes: When the budget is very small relative to the problem dimensionality, the algorithm may end prematurely far from the optimum. Highly multi‑modal landscapes can also lead to early convergence if restarts are insufficient.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Minimal black‑box minimization algorithm for the GNBG benchmark.
    Implements a two‑phase approach: random sampling followed by adaptive hill climbing.
    """

    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the problem (number of decision variables).
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the algorithm on the given black‑box function.

        Parameters
        ----------
        func : callable
            Objective function to minimize. Expected to accept a 1‑D numpy array
            and return a scalar value. Bounds are read from either func.lower/upper
            or func.bounds.lb/func.bounds.ub.

        Returns
        -------
        best_x : np.ndarray
            Best solution found.
        best_y : float
            Objective value at best_x.
        """
        lower, upper = self._get_bounds(func)

        # If the budget is too small for a meaningful search, just sample uniformly.
        if self.budget <= self.dim:
            return self._random_sample(lower, upper, func)

        # ----- Phase 1: initial random sampling -----
        # Use about 20% of the budget for sampling, capped by dim*5 points.
        n_sample = min(int(self.budget * 0.2) + 1, self.dim * 5, self.budget - 1)
        evals = 0
        best_x = None
        best_y = np.inf

        for _ in range(n_sample):
            x = lower + np.random.rand(self.dim) * (upper - lower)
            y = float(func(x))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            if evals >= self.budget:
                return best_x, best_y

        # ----- Phase 2: hill climbing with adaptive step size -----
        x = best_x.copy()
        y = best_y
        # Initial step: 10% of the bound range, never zero.
        step = (upper - lower) * 0.1
        step = np.maximum(step, 1e-6)

        fail_count = 0
        max_fail = 5  # shrink step after this many consecutive failures

        while evals < self.budget:
            # Generate a candidate via Gaussian perturbation.
            d = np.random.randn(self.dim)
            x_new = np.clip(x + step * d, lower, upper)
            y_new = float(func(x_new))
            evals += 1

            # Update global best if improved.
            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()

            # Accept move if it improves the current point.
            if y_new < y:
                x = x_new
                y = y_new
                fail_count = 0
                # Grow step on success, but keep it bounded.
                step = np.minimum(step * 1.1, (upper - lower) * 0.5)
            else:
                fail_count += 1
                if fail_count >= max_fail:
                    step *= 0.5
                    fail_count = 0
                    # If step becomes negligible, restart from a random point.
                    if np.all(step < (upper - lower) * 1e-4):
                        x = lower + np.random.rand(self.dim) * (upper - lower)
                        y = float(func(x))
                        evals += 1
                        if y < best_y:
                            best_y = y
                            best_x = x.copy()
                        step = (upper - lower) * 0.1
                        if evals >= self.budget:
                            break

        return best_x, best_y

    def _random_sample(self, lower, upper, func):
        """Simple uniform sampling when budget is very small."""
        best_x = None
        best_y = np.inf
        for _ in range(self.budget):
            x = lower + np.random.rand(self.dim) * (upper - lower)
            y = float(func(x))
            if y < best_y:
                best_y = y
                best_x = x.copy()
        return best_x, best_y

    def _get_bounds(self, func):
        """
        Extract problem bounds from the function object.

        Supports:
        - func.lower / func.upper
        - func.bounds.lb / func.bounds.ub
        - func.bounds as a (lb, ub) tuple

        If none are present, defaults to [0, 1]^dim.
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            bounds = func.bounds
            if hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
                lower = np.asarray(bounds.lb, dtype=float)
                upper = np.asarray(bounds.ub, dtype=float)
            else:
                # Assume bounds is a tuple/list of (lb, ub)
                lb, ub = bounds
                lower = np.asarray(lb, dtype=float)
                upper = np.asarray(ub, dtype=float)
        else:
            lower = np.zeros(self.dim)
            upper = np.ones(self.dim)

        # Ensure lower <= upper
        if np.any(lower > upper):
            lower, upper = upper, lower

        return lower, upper
