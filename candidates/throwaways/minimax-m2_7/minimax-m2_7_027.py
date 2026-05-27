# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple Differential Evolution (DE) algorithm for bound‑constrained black‑box minimization. DE maintains a population of candidate solutions, creates new candidates by mutating random individuals and recombining them with the target, then selects the better of the two.
# Search state: The algorithm stores a population of vectors, their objective values, and the best solution found so far.
# Candidate generation: For each target vector a donor is formed by taking a random individual and adding a scaled difference of two other random individuals (mutation factor F). A trial vector is then created by binomial crossover between the donor and the target.
# Selection and replacement: The trial vector competes directly with its target; if the trial has a lower objective value it replaces the target in the next generation.
# Adaptation: No explicit adaptation of F or crossover rate CR; they are fixed (F=0.5, CR=0.5). Population size is chosen relative to dimension and budget.
# Exploration mechanisms: Random selection of base and differential vectors provides broad exploration; the fixed mutation strength encourages searching new regions.
# Exploitation mechanisms: Selection pressure drives the population toward better solutions; elite best solution is tracked across generations.
# Boundary handling: All vectors are clipped to the provided bounds after mutation and crossover to stay feasible.
# Budget strategy: The algorithm evaluates the objective exactly the number of times specified by the budget and terminates when the budget is exhausted, never performing extra evaluations.
# Closest known influences: Classic Differential Evolution (DE/rand/1/bin) as described by Storn and Price (1997).
# Novelty or unusual aspects: Simplified DE with a small fixed population size to keep the per‑evaluation overhead low; designed to be a compact, readable baseline.
# Failure modes: May converge prematurely on highly multi‑modal landscapes if the population size is too small relative to the problem difficulty; fixed control parameters may not be optimal for all functions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple Differential Evolution (DE) for bound‑constrained black‑box minimization.

    The algorithm maintains a population of candidate solutions. In each iteration,
    for every target vector a donor is created by mutating a random individual with
    the scaled difference of two other random individuals. A trial vector is formed
    by binomial crossover between the donor and the target. The trial replaces the
    target if it has a lower objective value. The process repeats until the evaluation
    budget is exhausted.

    Parameters
    ----------
    budget : int
        Maximum number of objective function evaluations allowed.
    dim : int
        Dimensionality of the problem (number of decision variables).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # Choose a population size that is at least 4 (to allow three distinct
        # mutation vectors) and bounded by the budget and a reasonable maximum.
        # The factor 5*dim gives a moderate scaling with dimension, capped at 200.
        self.popsize = min(max(4, 5 * dim), budget, 200)

        # Fixed DE control parameters
        self.F = 0.5       # Mutation scaling factor
        self.CR = 0.5      # Crossover probability

    def __call__(self, func):
        """
        Run the DE optimizer on the given black‑box objective function.

        Parameters
        ----------
        func : callable
            A function that takes a 1‑D numpy array of shape (dim,) and returns a scalar.
            It must expose either ``lower``/``upper`` attributes or a ``bounds`` object
            with ``lb``/``ub`` attributes that define the search domain.

        Returns
        -------
        best_x : numpy.ndarray
            The decision vector that achieved the smallest objective value.
        best_y : float
            The objective value at ``best_x``.
        """
        # ------------------------------------------------------------------
        # Detect search bounds (support two common interfaces)
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: assume a modest symmetric domain if nothing is provided
            lb = np.full(self.dim, -10.0)
            ub = np.full(self.dim, 10.0)

        # Ensure lb/ub are proper 1‑D arrays of length dim
        lb = np.atleast_1d(lb)
        ub = np.atleast_1d(ub)
        if lb.shape[0] == 1:
            lb = np.full(self.dim, lb[0])
        if ub.shape[0] == 1:
            ub = np.full(self.dim, ub[0])

        # ------------------------------------------------------------------
        # Initialise population uniformly within the bounds
        # ------------------------------------------------------------------
        pop = np.random.uniform(lb, ub, size=(self.popsize, self.dim))

        # Evaluate initial population
        fitness = np.full(self.popsize, np.inf, dtype=float)
        evals = 0
        best_x = None
        best_y = np.inf

        for i in range(self.popsize):
            if evals >= self.budget:
                break
            y = func(pop[i])
            fitness[i] = y
            evals += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # ------------------------------------------------------------------
        # Main evolution loop: generate, evaluate, and select
        # ------------------------------------------------------------------
