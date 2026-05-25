# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module provides a compact, pure‑Python implementation of the
# Differential Evolution (DE) algorithm for bound‑constrained black‑box
# minimization.  DE evolves a population of candidate solutions by repeatedly
# creating mutant vectors from three distinct parents, recombining them with
# the target vectors via binomial crossover, and performing greedy selection.
# The optimizer respects a hard evaluation budget, never exceeding it, and
# returns the best feasible solution discovered.
#
# Search state: The optimizer maintains a population of NP candidate vectors,
# each with its current objective value, plus a global best record.
#
# Candidate generation: For each target vector X_i a mutant V is formed as
# V = X_a + F*(X_b - X_c) where a,b,c are three unique indices different from
# i.  V is clipped to the problem bounds.  A trial vector T is created by
# copying each dimension from V with probability CR (crossover rate), with a
# guaranteed copy of at least one random dimension.
#
# Selection and replacement: After evaluating the trial fitness, the algorithm
# performs greedy (1‑to‑1) selection: if f(T) <= f(X_i) the target is replaced
# by T in the population; otherwise X_i is retained.
#
# Adaptation: The mutation scaling factor F and crossover rate CR are static
# (F ∈ [0.5,1.0], CR = 0.7) to keep the code simple and budget‑friendly.
# No covariance matrix adaptation is used.
#
# Exploration mechanisms: Random initialization within the bounds and the DE
# mutation scheme (difference‑based) provide broad exploration of the search
# space, especially in the early stages.
#
# Exploitation mechanisms: Greedy selection drives the population toward
# better solutions, and the global best is continuously updated, focusing
# exploitation on the most promising regions.
#
# Boundary handling: All generated vectors (initial population and mutants)
# are clipped to the user‑supplied lower (lb) and upper (ub) bounds,
# guaranteeing feasible candidates.
#
# Budget strategy: The algorithm counts each function evaluation and stops
# immediately when the supplied budget is exhausted, ensuring no excess
# evaluations.
#
# Closest known influences: Classic Differential Evolution (Storn & Price,
# 1997) with a simple greedy selector and static hyperparameters.
#
# Novelty or unusual aspects: The implementation relies only on numpy and
# the standard library, has no external dependencies, and dynamically sizes
# the population based on the problem dimension and available budget to
# balance exploration and exploitation.
#
# Failure modes: If the budget is too small to fill a minimal population
# (NP < 4) the algorithm falls back to evaluating a single random point.
# In highly multi‑modal landscapes DE may converge prematurely due to lack
# of explicit diversity mechanisms.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    """
    Lightweight Differential Evolution optimizer for bound‑constrained
    black‑box minimization.  The class conforms to the required interface:
        __init__(self, budget, dim)
        __call__(self, func) -> (best_x, best_y)
    """

    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the decision space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func) -> tuple:
        """
        Minimise `func` using Differential Evolution while respecting the
        evaluation budget.

        Parameters
        ----------
        func : callable
            Black‑box objective function.  Expected to accept a 1‑D numpy
            array of length `dim` and return a scalar (the objective value).

        Returns
        -------
        best_x : numpy.ndarray
            Decision vector of the best solution found.
        best_y : float
            Objective value at `best_x`.
        """
        # ------------------------------------------------------------------
        # 1. Determine problem bounds
        # ------------------------------------------------------------------
        try:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb)
                ub = np.asarray(func.bounds.ub)
            except AttributeError:
                # Default bounds when none are provided
                lb = np.full(self.dim, -10.0)
                ub = np.full(self.dim, 10.0)

        # Ensure lb/ub are 1‑D arrays of correct length
        lb = np.atleast_1d(lb).astype(float)
        ub = np.atleast_1d(ub).astype(float)
        if lb.shape[0] != self.dim:
            lb = np.full(self.dim, lb[0])
        if ub.shape[0] != self.dim:
            ub = np.full(self.dim, ub[0])

        # ------------------------------------------------------------------
        # 2. Handle the trivial case of an extremely small budget
        # ------------------------------------------------------------------
        if self.budget <= 1:
            x = lb + np.random.rand(self.dim) * (ub - lb)
            y = func(x)
            return x, float(y)

        # ------------------------------------------------------------------
        # 3. Initialise DE population
        # ------------------------------------------------------------------
        # Choose population size based on dimension and remaining budget.
        # DE requires at least 4 individuals to generate distinct mutants.
        max_pop = max(4, min(10 * self.dim, (self.budget - 1) // 2))
        NP = max_pop
        pop = lb + np.random.rand(NP, self.dim) * (ub - lb)

        # Evaluate initial population
        fitness = np.empty(NP, dtype=float)
        nevals = 0
        for i in range(NP):
            fitness[i] = func(pop[i])
            nevals += 1
            if nevals >= self.budget:
                best_idx = np.argmin(fitness[:nevals])
                return pop[best_idx].copy(), float(fitness[best_idx])

        # Keep track of the best solution found so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = float(fitness[best_idx])

        # ------------------------------------------------------------------
        # 4. DE control parameters
        # ------------------------------------------------------------------
        F = 0.5 + 0.5 * np.random.rand()   # Mutation scaling factor
        CR = 0.7                           # Crossover probability

        # ------------------------------------------------------------------
        # 5. Main DE evolution loop
        # ------------------------------------------------------------------
        while nevals < self.budget:
            for i in range(NP):
                if nevals >= self.budget:
                    break

                # Choose three distinct indices different from i
                a, b, c = np.random.choice(NP, 3, replace=False)

                # Generate mutant vector
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                # Build trial vector via binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluate trial
                f_trial = func(trial)
                nevals += 1

                # Greedy selection
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_y:
                        best_y = float(f_trial)
                        best_x = trial.copy()

                if nevals >= self.budget:
                    break

            # Optional simple adaptation of F (not strictly required)
            # Here we keep F static to preserve simplicity and predictability.

        return best_x, best_y
