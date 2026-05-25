# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A minimal CMA‑ES‑style evolution strategy that maintains a search distribution
# (mean and isotropic step‑size) and adapts the step‑size using a simple 1/5‑th success rule.
# Search state: The algorithm keeps a current mean vector (center of the sampling distribution)
# and a scalar step‑size sigma controlling the spread of candidate solutions.
# Candidate generation: At each generation, lambda individuals are sampled from a normal
# distribution N(mean, sigma²I).  Samples are clipped to the problem’s bounds.
# Selection and replacement: The mu best individuals (lowest function values) are retained.
# A weighted recombination of these individuals forms the new mean, biasing the search toward
# promising regions.
# Adaptation: After each generation the step‑size sigma is increased by a factor of 1.2 if the
# best fitness improved (success), otherwise decreased by 0.8 (failure).  sigma is also
# constrained to stay within a safe fraction of the bound range.
# Exploration mechanisms: A relatively large initial sigma encourages broad exploration,
# and random sampling each generation injects diversity.
# Exploitation mechanisms: Recombination of the best individuals concentrates the search
# around the currently known optimum.
# Boundary handling: All generated points are hard‑clipped to the lower and upper bounds
# provided by the benchmark.  sigma is kept within a modest range to avoid stepping outside
# the feasible region.
# Budget strategy: The algorithm counts each function evaluation and terminates as soon as
# the evaluation counter reaches the supplied budget.  The very first point (the centre of
# the box) is evaluated before the generational loop to guarantee at least one evaluation.
# Closest known influences: This is a stripped‑down version of the Covariance Matrix
# Adaptation Evolution Strategy (CMA‑ES) using only a diagonal covariance (separate
# variance per coordinate) and a basic step‑size adaptation scheme, similar to the classic
# 1/5‑th rule used in early evolution strategies.
# Novelty or unusual aspects: By deliberately limiting the covariance to a scalar times the
# identity, the algorithm stays extremely lightweight and scales linearly with dimension,
# making it suitable for high‑dimensional black‑box problems while still retaining the core
# self‑adaptation idea of CMA‑ES.
# Failure modes: If the evaluation budget is smaller than the dimensionality (e.g., less than
# a few dozen evaluations), the algorithm may not have enough samples to improve over random
# sampling.  Tight bounds can cause clipping that slightly distorts the sampling distribution.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initialise the optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the problem (number of variables).
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.evaluations = 0  # optional counter, not required by the interface

    def __call__(self, func):
        """
        Run the optimizer on the given black‑box function.

        Parameters
        ----------
        func : callable
            A function that takes a 1‑D numpy array of length ``dim`` and returns
            a scalar (the objective value).  The function must expose the bounds
            either as ``func.lower`` / ``func.upper`` or as ``func.bounds.lb`` /
            ``func.bounds.ub``.

        Returns
        -------
        best_x : numpy.ndarray
            The best (lowest‑value) solution found.
        best_y : float
            The objective value of ``best_x``.
        """
        # ------------------------------------------------------------------
        # 1. Obtain problem bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'lower'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # assume func.bounds has attributes lb and ub
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: use a symmetric box of -10 … 10 for every dimension
            lb = np.full(self.dim, -10.0)
            ub = np.full(self.dim, 10.0)

        # Ensure the bounds are proper vectors
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        # ------------------------------------------------------------------
        # 2. Initialise the search distribution (mean and step‑size)
        # ------------------------------------------------------------------
        # start at the centre of the box
        mean = (lb + ub) * 0.5
        # initial step‑size is one sixth of the range (≈covers 99 % of a normal)
        sigma = (ub - lb) / 6.0

        # ------------------------------------------------------------------
        # 3. Population size (λ) and number of parents (μ)
        # ------------------------------------------------------------------
        # λ follows the common CMA‑ES default: 4 + floor(3·log(dim))
        lam = max(4, int(4 + 3 * np.log(self.dim + 1)))
        mu = lam // 2   # select the best half

        # ------------------------------------------------------------------
        # 4. Recombination weights for the selected parents
        # ------------------------------------------------------------------
        # using soft‑max (log‑like) weights so that the best parent gets the
        # largest weight and the others get decreasing weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1) + 1)
        weights /= weights.sum()   # normalise so that they sum to one

        # ------------------------------------------------------------------
        # 5. Evaluation counter and initial incumbent evaluation
        # ------------------------------------------------------------------
        evals = 0
        if self.budget <= 0:
            # nothing can be evaluated – return a random feasible point
            x0 = np.random.uniform(lb, ub)
            return x0, func(x0)

        # evaluate the starting point (the centre of the box)
        best_x = mean.copy()
        best_y = func(best_x)
        evals += 1

        # ------------------------------------------------------------------
        # 6. Main CMA‑ES‑like loop (generation after generation)
        # ------------------------------------------------------------------
        while evals < self.budget:
            # How many individuals can we afford in this generation?
            remaining = self.budget - evals
            if remaining < lam:
                # Not enough budget for a full population – evaluate a smaller subset.
                lam_cur = remaining
                mu_cur = max(1, lam_cur // 2)
                # recompute weights for the smaller μ (optional but clean)
                weights_cur = np.log(mu_cur + 1) - np.log(np.arange(1, mu_cur + 1) + 1)
                weights_cur /= weights_cur.sum()
            else:
                lam_cur = lam
                mu_cur = mu
                weights_cur = weights

            # ----------------------------------------------------------------
            # 6.1 Sample λ candidate solutions from the current distribution
            # ----------------------------------------------------------------
            # With a diagonal covariance we can sample each coordinate independently:
            # mean + sigma * N(0,1)
            candidates = mean + sigma * np.random.randn(lam_cur, self.dim)
            # keep the samples inside the feasible region (hard boundaries)
            candidates = np.clip(candidates, lb, ub)

            # ----------------------------------------------------------------
            # 6.2 Evaluate the candidates (as long as we have budget)
            # ----------------------------------------------------------------
            fitness = np.empty(lam_cur)
            for i in range(lam_cur):
                if evals >= self.budget:
                    # budget exhausted – truncate the population
                    fitness = fitness[:i]
                    candidates = candidates[:i]
                    break
                fitness[i] = func(candidates[i])
                evals += 1

            # If we exhausted the budget while filling the population,
            # we still need to determine the best point seen so far.
            if evals >= self.budget and fitness.size == 0:
                break   # no new point was evaluated

            # ----------------------------------------------------------------
            # 6.3 Selection – keep the μ best individuals
            # ----------------------------------------------------------------
            # argsort gives indices that sort from smallest to largest (minimisation)
            sorted_idx = np.argsort(fitness)
            top_idx = sorted_idx[:mu_cur]

            # ----------------------------------------------------------------
            # 6.4 Weighted recombination to update the distribution mean
            # ----------------------------------------------------------------
            new_mean = np.dot(weights_cur, candidates[top_idx])  # shape (dim,)

            # ----------------------------------------------------------------
            # 6.5 Simple step‑size adaptation (1/5‑th success rule)
            # ----------------------------------------------------------------
            # If the best fitness in this generation is better than the overall best,
            # we consider the generation successful → increase σ.
            # Otherwise we shrink σ.
            gen_best_f = fitness[sorted_idx[0]]
            if gen_best_f < best_y:
                # success – explore more
                sigma *= 1.2
                best_y = gen_best_f
                best_x = candidates[sorted_idx[0]].copy()
            else:
                # no improvement – be more conservative
                sigma *= 0.8

            # keep σ within a safe range (avoid vanishing or exploding)
            sigma = np.clip(sigma, (ub - lb) * 1e-3, (ub - lb) * 0.5)

            # ----------------------------------------------------------------
            # 6.6 Update the distribution mean for the next generation
            # ----------------------------------------------------------------
            mean = new_mean

        # ------------------------------------------------------------------
        # 7. Return the best solution discovered
        # ------------------------------------------------------------------
        return best_x, best_y
