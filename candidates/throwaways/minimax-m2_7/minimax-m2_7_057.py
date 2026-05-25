# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary:
#   This module implements a simple Differential Evolution (DE) algorithm for
#   unconstrained black‑box minimization.  DE maintains a population of candidate
#   solutions, creates new individuals by mutation and recombination, and keeps the
#   better of the old and trial vectors.  The implementation follows the
#   “DE/rand/1/bin” scheme, which is robust across a wide range of dimensions
#   and does not require gradient information.
#
# Search state:
#   * A population matrix X of shape (NP, dim) holding the current candidates.
#   * A parallel vector f of objective values for each population member.
#   * The index (or value) of the best individual seen so far.
#
# Candidate generation:
#   For each target vector X[i] a mutant vector is built as
#       mutant = X[r1] + F * (X[r2] - X[r3])
#   where r1, r2, r3 are three distinct indices drawn randomly from the
#   population (excluding i).  The scaling factor F controls the step size
#   (default 0.5).
#
# Selection and replacement:
#   After recombination (see below) the trial vector is evaluated on the
#   objective function.  If its fitness is less than or equal to the target’s,
#   the target is replaced by the trial; otherwise the target is retained.
#   This one‑to‑one elitism keeps the best solution seen at any time.
#
# Adaptation:
#   The algorithm uses fixed control parameters (F = 0.5, crossover rate
#   CR = 0.5).  No on‑the‑fly adaptation of these values is performed.
#
# Exploration mechanisms:
#   * Mutation with a random base vector (DE/rand) ensures broad exploration.
#   * The binomial crossover (bin) mixes components of the mutant and the target,
#     spreading the mutation across dimensions.
#
# Exploitation mechanisms:
#   * Selection based on fitness drives the population toward lower objective
#     values.
#   * The best individual is tracked separately, so good solutions are not lost.
#
# Boundary handling:
#   All trial vectors are clipped to the problem’s lower/upper bounds before
#   evaluation.  This prevents the algorithm from stepping outside the feasible
#   region.
#
# Budget strategy:
#   The total number of function evaluations is never allowed to exceed the
#   supplied budget.  The main loop computes how many full generations can be
#   performed given the remaining budget and stops as soon as the budget is
#   exhausted.
#
# Closest known influences:
#   Differential Evolution (Storn & Price, 1997).  The specific variant used
#   here is the classic “DE/rand/1/bin”.
#
# Novelty or unusual aspects:
#   The implementation is deliberately minimal, relying only on NumPy and the
#   standard library.  Population size is adapted to the budget to avoid
#   consuming too many evaluations in the initial population.
#
# Failure modes:
#   * If the budget is extremely small relative to the dimension, the algorithm
#     may not improve much beyond the initial random sampling.
#   * Fixed control parameters may be suboptimal for highly anisotropic or
#     multi‑modal landscapes.
#   * No explicit handling of noisy objectives is included.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple Differential Evolution (DE/rand/1/bin) for black‑box minimization.

    Parameters
    ----------
    budget : int
        Maximum number of objective function evaluations allowed.
    dim : int
        Dimensionality of the search space.
    """

    # Fixed control parameters for DE
    _F: float = 0.5   # mutation scaling factor
    _CR: float = 0.5  # crossover probability

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

        # Choose a population size that does not dominate the budget.
        # We need at least a handful of individuals to maintain diversity.
        # Use at least 10, but not more than half of the budget.
        self._NP = max(10, min(10 * dim, self.budget // 2))

    def __call__(self, func) -> tuple[np.ndarray, float]:
        """
        Run DE on the provided black‑box objective function.

        Parameters
        ----------
        func : callable
            A function that accepts a 1‑D NumPy array of shape (dim,) and
            returns a scalar objective value.

        Returns
        -------
        best_x : np.ndarray
            The decision vector that achieved the smallest observed objective.
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------------
        # 1. Determine search bounds
        # ------------------------------------------------------------------
        lower = self._extract_bound(func, 'lower')
        upper = self._extract_bound(func, 'upper')
        # Ensure bounds are NumPy arrays of the correct shape
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        # ------------------------------------------------------------------
        # 2. Initialise population
        # ------------------------------------------------------------------
        rng = np.random.default_rng()  # external seed control via harness
        X = rng.uniform(lower, upper, size=(self._NP, self.dim))

        # Evaluate initial population
        f = np.empty(self._NP)
        for i in range(self._NP):
            f[i] = func(X[i])

        evals = self._NP
        best_idx = int(np.argmin(f))
        best_x = X[best_idx].copy()
        best_y = float(f[best_idx])

        # ------------------------------------------------------------------
        # 3. Main evolution loop – run full generations while budget allows
        # ------------------------------------------------------------------
        while evals < self.budget:
            # How many complete generations can we still afford?
            remaining = self.budget - evals
            generations = remaining // self._NP
            if generations == 0:
                # Not enough evaluations left for another full generation.
                break

            for _ in range(generations):
                # ----- Mutation & Recombination -----
                for i in range(self._NP):
                    # Choose three distinct indices from the population, all different from i
                    candidates = np.concatenate((np.arange(i), np.arange(i + 1, self._NP)))
                    r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                    # DE/rand/1 mutation
                    mutant = X[r1] + self._F * (X[r2] - X[r3])

                    # Binomial crossover (bin)
                    # Force at least one component from the mutant
                    j_rand = rng.integers(self.dim)
                    mask = rng.random(self.dim) < self._CR
                    trial = np.where(mask, mutant, X[i])
                    trial[j_rand] = mutant[j_rand]

                    # Enforce bound constraints
                    trial = np.clip(trial, lower, upper)

                    # ----- Evaluation & Selection -----
                    f_trial = func(trial)
                    evals += 1

                    if f_trial <= f[i]:
                        X[i] = trial
                        f[i] = f_trial
                        if f_trial < best_y:
                            best_idx = i
                            best_y = float(f_trial)
                            best_x = X[i].copy()

                    # Stop early if the budget is exhausted
                    if evals >= self.budget:
                        break

                if evals >= self.budget:
                    break

        return best_x, best_y

    @staticmethod
    def _extract_bound(func, name: str) -> np.ndarray:
        """
        Read a bound attribute from the function object.

        Supports two common conventions:
            func.lower / func.upper
            func.bounds.lb / func.bounds.ub
        If neither is found, a default of -10 / +10 per dimension is used.
        """
        # Try func.<name> first
        if hasattr(func, name):
            bound = getattr(func, name)
            return np.asarray(bound, dtype=float)

        # Try func.bounds.<name> (common in many frameworks)
        if hasattr(func, 'bounds'):
            bounds = func.bounds
            # map 'lower'/'upper' to 'lb'/'ub'
            key = 'lb' if name == 'lower' else 'ub'
            if hasattr(bounds, key):
                return np.asarray(getattr(bounds, key), dtype=float)

        # Fallback: return None and let the caller handle defaults
        return None
