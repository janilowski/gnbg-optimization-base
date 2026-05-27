import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: The algorithm combines uniform random sampling (exploration) with a stochastic hill‑climbing (exploitation) phase. It first spends a small fraction of the evaluation budget probing the search space uniformly at random, keeps the best point found, and then repeatedly mutates that point with Gaussian noise. Better mutants are accepted, causing the search to move downhill; when a series of proposals fail the step‑size (sigma) is shrunk to focus the search. Occasionally the algorithm injects a completely new random candidate to escape local minima.
# Search state: The algorithm maintains the best feasible point (best_x) and its objective value (best_y), a mutable step‑size vector sigma, and a counter of consecutive failures to adapt sigma.
# Candidate generation: In the exploration phase candidates are drawn uniformly from the user‑provided bounds. In the exploitation phase candidates are generated as best_x + sigma·N(0,1) and clipped back into the bounds.
# Selection and replacement: A candidate replaces the current best only if its objective value is strictly smaller (minimization). Otherwise the failure counter increments.
# Adaptation: sigma is increased slightly (×1.1) after a successful step and multiplied by 0.5 after a predefined number of consecutive failures (shrink). This balances exploration and exploitation.
# Exploration mechanisms: Initial random sampling and occasional random restarts provide global diversification.
# Exploitation mechanisms: Local hill‑climbing with Gaussian perturbations drives the search toward lower objective values near the current best.
# Boundary handling: All generated points are clipped to the [lower, upper] bounds to keep the search feasible.
# Budget strategy: The algorithm divides the total evaluation budget into a random‑sampling portion (≈30 % of the budget) and a hill‑climbing portion (remaining evaluations). It never performs more evaluations than allowed.
# Closest known influences: The method resembles a simple Evolution Strategy (ES) with (1+1) mutate‑and‑select scheme combined with a random‑initialisation phase, reminiscent of Random Search + Hill Climbing hybrids.
# Novelty or unusual aspects: The step‑size vector is per‑dimension, tied to the side‑length of the search space, and sigma is adapted dynamically based on success/failure counts. Random restarts are performed with a low probability to avoid stagnation.
# Failure modes: If the landscape is highly multi‑modal and the random phase misses the basin of attraction of the global optimum, the hill‑climber may converge to a local minimum. The algorithm also relies on Gaussian noise, which may be inefficient for very anisotropic landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """
    A simple hybrid optimizer: random sampling followed by stochastic hill climbing.
    Works with any black‑box function that accepts a 1‑D numpy array and provides
    bound information via either `func.lower/func.upper` or `func.bounds.lb/ub`.
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Execute the optimization.

        Parameters
        ----------
        func : callable
            Objective function to minimize. Must accept a 1‑D numpy array
            of length ``dim`` and return a scalar. It must also expose
            either ``func.lower``/``func.upper`` or ``func.bounds.lb``/
            ``func.bounds.ub`` providing per‑dimension lower and upper bounds.

        Returns
        -------
        best_x : np.ndarray
            Best feasible solution found.
        best_y : float
            Corresponding objective value.
        """
        # ------------------------------------------------------------------
        # Retrieve bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: use a large symmetric box if no bounds are provided.
            lower = np.full(self.dim, -100.0)
            upper = np.full(self.dim, 100.0)

        # Ensure lower/upper are 1‑D arrays of length dim
        if lower.shape != (self.dim,):
            lower = np.full(self.dim, float(lower.flat[0]))
            upper = np.full(self.dim, float(upper.flat[0]))

        # ------------------------------------------------------------------
        # Initialisation
        # ------------------------------------------------------------------
        best_x = None
        best_y = np.inf
        evals = 0

        # ------------------------------------------------------------------
        # Phase 1 – Random sampling (exploration)
        # ------------------------------------------------------------------
        # Spend roughly 30 % of the budget on uniform random search.
        random_budget = max(int(self.budget * 0.3), 1)
        for _ in range(random_budget):
            x = lower + np.random.rand(self.dim) * (upper - lower)
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            if evals >= self.budget:
                break

        # ------------------------------------------------------------------
        # Phase 2 – Stochastic hill climbing (exploitation)
        # ------------------------------------------------------------------
        if evals < self.budget and best_x is not None:
            # Initial step‑size proportional to the side length of the domain.
            sigma = 0.2 * (upper - lower)
            sigma = np.maximum(sigma, 1e-12)  # avoid degenerate zero width

            fail_count = 0          # consecutive failures
            max_fail = 5            # shrink sigma after this many failures

            while evals < self.budget:
                # Generate a mutant by adding Gaussian noise.
                candidate = best_x + sigma * np.random.randn(self.dim)
                # Keep candidate within bounds.
                candidate = np.clip(candidate, lower, upper)

                y_candidate = func(candidate)
                evals += 1

                if y_candidate < best_y:
                    # Accept improvement
                    best_y = y_candidate
                    best_x = candidate.copy()
                    fail_count = 0
                    # Slightly increase step‑size to encourage broader search.
                    sigma = sigma * 1.1
                else:
                    fail_count += 1
                    if fail_count >= max_fail:
                        sigma = sigma * 0.5
                        fail_count = 0

                # Random restart with low probability to escape local minima.
                if np.random.rand() < 0.05 and evals < self.budget:
                    x = lower + np.random.rand(self.dim) * (upper - lower)
                    y = func(x)
                    evals += 1
                    if y < best_y:
                        best_y = y
                        best_x = x.copy()
                    # Reset step‑size after a restart.
                    sigma = 0.2 * (upper - lower)
                    sigma = np.maximum(sigma, 1e-12)
                    fail_count = 0

        # If budget is exhausted before any evaluation occurs (should not happen),
        # return a random feasible point.
        if best_x is None:
            best_x = lower + np.random.rand(self.dim) * (upper - lower)
            best_y = func(best_x)

        return best_x, best_y
