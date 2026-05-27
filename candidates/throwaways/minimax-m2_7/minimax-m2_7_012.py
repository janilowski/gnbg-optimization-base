# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module provides a simple Differential Evolution (DE) optimizer tailored
# for bound‑constrained black‑box minimization on the GNBG benchmark suite. The algorithm
# evolves a population of candidate solutions using mutation, recombination and greedy
# selection while respecting a supplied evaluation budget.
#
# Search state:
#   - The optimizer maintains a population matrix (pop_size × dim). Each row is a
#     candidate point. A parallel vector stores the corresponding fitness values.
#   - The best observed point and its fitness are tracked continuously.
#
# Candidate generation:
#   - Offspring are produced with the classic DE/rand/1 mutation scheme:
#       mutant = pop[r0] + F * (pop[r1] - pop[r2])
#     where r0, r1, r2 are distinct indices different from the target index.
#   - Binomial crossover generates a trial vector: each gene is taken from the mutant
#     with probability CR, otherwise from the target vector.
#   - All trial vectors are clipped to the problem bounds to enforce feasibility.
#
# Selection and replacement:
#   - After evaluating the trial vectors, the algorithm performs a greedy selection:
#     if the trial fitness is ≤ the target fitness, the target is replaced by the trial;
#     otherwise the target is kept. This yields a (μ+λ) strategy with μ=λ=pop_size.
#   - The global best is updated whenever a better individual is found.
#
# Adaptation:
#   - The scaling factor F and crossover rate CR are set to sensible defaults (0.8 and
#     0.9 respectively). They are not adapted during the run to keep the implementation
#     simple and deterministic.
#
# Exploration vs. exploitation:
#   - The random base vector in mutation encourages exploration, while the crossover
#     blends target and mutant information to fine‑tune solutions. Greedy selection
#     retains the best candidates, thus balancing exploitation.
#
# Boundary handling:
#   - After mutation and before recombination, the mutant vector is clipped to the
#     lower/upper bounds. Any component that falls outside the allowed range is
#     projected back onto the closest bound.
#
# Budget strategy:
#   - The optimizer counts function evaluations and stops as soon as the remaining
#     budget is insufficient for a full generation (i.e., eval_count + pop_size > budget).
#   - If the initial budget is smaller than the chosen population size, a pure random
#     search is performed for the available evaluations.
#
# Closest known influences:
#   - Classic Differential Evolution (Storn & Price, 1997) with the rand/1/bin scheme.
#   - The implementation follows the standard DE flow without self‑adaptation (e.g., jDE).
#
# Novelty or unusual aspects:
#   - No unusual aspects; the code is deliberately straightforward to demonstrate a
#     clear mapping between DE components and the required benchmark interface.
#
# Failure modes:
#   - If the problem is highly multi‑modal and the budget is very low, the optimizer
#     may converge prematurely to a local optimum due to insufficient diversity.
#   - The fixed F and CR may be suboptimal for certain landscapes; however, they
#     provide a robust baseline for a wide range of GNBG instances.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple Differential Evolution optimizer for bound‑constrained black‑box problems.
    """

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
        # Population size heuristics: 10*dim but at least 50, and not larger than budget.
        self.pop_size = min(max(10 * dim, 50), budget)
        # DE control parameters (fixed for simplicity)
        self.F = 0.8   # scaling factor
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        """
        Optimize ``func`` within the supplied evaluation budget.

        Parameters
        ----------
        func : callable
            Black‑box objective function. It must accept a 1‑D NumPy array of length ``dim``
            and return a scalar. The function should expose either ``lower``/``upper``
            attributes or a ``bounds`` attribute with ``lb``/``ub`` to specify the search
            range.

        Returns
        -------
        best_x : np.ndarray
            Best solution found (vector of length ``dim``).
        best_y : float
            Objective value at ``best_x``.
        """
        # ------------------------------------------------------------------
        # Determine bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # No bounds provided – fall back to a huge hyper‑cube.
            lower = np.full(self.dim, -1e9, dtype=float)
            upper = np.full(self.dim, 1e9, dtype=float)

        # Ensure lower/upper are arrays of shape (dim,)
        lower = _broadcast_to_shape(lower, self.dim)
        upper = _broadcast_to_shape(upper, self.dim)

        # ------------------------------------------------------------------
        # Handle pathological case: no evaluations allowed
        # ------------------------------------------------------------------
        if self.pop_size == 0:
            # Return a zero vector as a placeholder.
            return np.zeros(self.dim, dtype=float), np.inf

        # ------------------------------------------------------------------
        # Initial population (random sampling)
        # ------------------------------------------------------------------
        pop = np.random.uniform(lower, upper, size=(self.pop_size, self.dim))
        eval_count = 0
        fitness = np.empty(self.pop_size, dtype=float)

        for i in range(self.pop_size):
            fitness[i] = func(pop[i])
            eval_count += 1

        # Track best solution found so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = float(fitness[best_idx])

        # If the budget is exhausted after the initial draw, return the best seen.
        if eval_count >= self.budget:
            return best_x, best_y

        # ------------------------------------------------------------------
        # Main DE loop
        # ------------------------------------------------------------------
        while True:
            # Check whether we have enough remaining evaluations for a full generation.
            if eval_count + self.pop_size > self.budget:
                break

            # ----- Generate trial vectors ---------------------------------
            trials = np.empty_like(pop)

            for i in range(self.pop_size):
                # Choose three distinct indices different from i
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, size=3, replace=False)

                # DE/rand/1 mutation
                mutant = pop[a] + self.F * (pop[b] - pop[c])

                # Clip to bounds
                mutant = np.clip(mutant, lower, upper)

                # Binomial crossover
                mask = np.random.random(self.dim) < self.CR
                # Guarantee at least one component from mutant
                if not mask.any():
                    mask[np.random.randint(0, self.dim)] = True

                trials[i] = np.where(mask, mutant, pop[i])

            # ----- Evaluate trial vectors ---------------------------------
            for i in range(self.pop_size):
                fitness_i = func(trials[i])
                eval_count += 1

                # Greedy selection
                if fitness_i <= fitness[i]:
                    pop[i] = trials[i]
                    fitness[i] = fitness_i
                    # Update global best if improved
                    if fitness_i < best_y:
                        best_x = pop[i].copy()
                        best_y = float(fitness_i)

        return best_x, best_y


def _broadcast_to_shape(arr, dim):
    """
    Ensure ``arr`` is a NumPy array of length ``dim``.
    If ``arr`` is a scalar or an array with fewer dimensions,
    broadcast it accordingly.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.shape == ():
        arr = np.full(dim, arr)
    elif arr.shape != (dim,):
        arr = np.broadcast_to(arr, (dim,)).copy()
    return arr
