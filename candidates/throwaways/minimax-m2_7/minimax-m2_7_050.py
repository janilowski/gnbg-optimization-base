# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a simple Differential Evolution (DE) optimizer adapted
# for low‑budget black‑box minimization. DE is a population‑based evolutionary algorithm
# that generates new candidate solutions by mutating existing ones and keeping the best.
# Search state: The algorithm maintains a population of `NP` candidate points and their
# objective values, plus a record of the globally best point found so far.
# Candidate generation: For each individual a mutant vector is formed by adding a scaled
# difference of two randomly chosen, distinct population members to a third member.
# Binomial crossover then mixes the mutant with the original individual, ensuring at
# least one dimension comes from the mutant.
# Selection and replacement: After evaluating the trial vector, it replaces the original
# individual only if it is not worse (minimization). The global best is updated whenever
# a better solution is discovered.
# Adaptation: The control parameters (scale factor F and crossover rate CR) are kept
# fixed; the algorithm relies on the population size to balance exploration and
# exploitation. Any remaining evaluation budget after the DE generations is spent on
# uniform random sampling to use up the full budget.
# Exploration mechanisms: Mutation with scaled differences introduces diversity, while
# the crossover operation recombines information from different individuals.
# Exploitation mechanisms: Selection pressure quickly propagates beneficial traits,
# and keeping the best point guarantees that the current optimum is never lost.
# Boundary handling: All generated points (initial population, mutants, trials) are
# clipped to the problem’s lower and upper bounds to stay within the feasible region.
# Budget strategy: The algorithm first spends `NP` evaluations on the initial
# population. Then it runs as many full DE generations as the budget allows (each
# generation costs `NP` evaluations). Any leftover evaluations are used for pure random
# search, guaranteeing that the total number of function calls never exceeds the
# supplied budget.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997) with
# the “rand/1/bin” scheme, but simplified to a single fixed population size for
# compactness.
# Novelty or unusual aspects: The implementation is deliberately minimal, using only
# NumPy and the standard library, and includes an explicit fallback to random search
# when the budget is too small to support a viable DE population.
# Failure modes: If the budget is extremely low relative to the dimensionality, the
# algorithm degrades to random search, which may perform poorly on rugged landscapes.
# The fixed DE parameters may also be suboptimal for certain function classes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple Differential Evolution (DE) optimizer for black‑box minimization.

    The optimizer respects the evaluation budget, reads problem bounds from the
    provided function object, and returns the best found solution and its objective
    value.
    """

    def __init__(self, budget: int, dim: int):
        """
        Initialize the optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the search space (number of variables).
        """
        self.budget = budget
        self.dim = dim

    def _get_bounds(self, func) -> tuple:
        """
        Extract lower and upper bounds from the function object.

        The function object may store bounds as attributes `lower`/`upper` or as
        `bounds.lb`/`bounds.ub`. If neither is present, default bounds are set to
        ``-inf`` / ``+inf`` for each dimension.

        Parameters
        ----------
        func : callable
            The objective function (black‑box).

        Returns
        -------
        lb, ub : numpy.ndarray
            Arrays of lower and upper bounds for each dimension.
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # No bounds provided – use unbounded limits.
            lb = np.full(self.dim, -np.inf)
            ub = np.full(self.dim, np.inf)
        # Ensure they are 1‑D arrays of length dim.
        lb = np.reshape(lb, -1)
        ub = np.reshape(ub, -1)
        return lb, ub

    def __call__(self, func) -> tuple:
        """
        Run the optimizer on the given objective function.

        Parameters
        ----------
        func : callable
            Objective function to be minimized. It must accept a NumPy array of
            shape ``(dim,)`` and return a scalar.

        Returns
        -------
        best_x : numpy.ndarray
            The best (lowest‑value) solution found.
        best_y : float
            The objective value corresponding to ``best_x``.
        """
        # -----------------------------------------------------------------
        # 1. Determine problem bounds.
        # -----------------------------------------------------------------
        lb, ub = self._get_bounds(func)

        # -----------------------------------------------------------------
        # 2. If the budget is too small to form a viable DE population,
        #    fall back to a pure random search.
        # -----------------------------------------------------------------
        if self.budget < 4:
            # Not enough evals for a minimal population; just sample randomly.
            best_x = None
            best_y = np.inf
            for _ in range(self.budget):
                x = np.random.uniform(lb, ub)
                y = func(x)
                if y < best_y:
                    best_x = x.copy()
                    best_y = y
            return best_x, best_y

        # -----------------------------------------------------------------
        # 3. Set up a DE population.
        #    Population size NP is chosen as a compromise between exploration
        #    and computational cost: at least 4 individuals (required for DE)
        #    and at most 5 * dim. The budget also limits the maximum feasible
        #    population size (budget // 2) because each generation costs NP
        #    evaluations.
        # -----------------------------------------------------------------
        NP = min(max(5 * self.dim, 4), max(self.budget // 2, 4))
        # Ensure NP does not exceed the budget.
        NP = min(NP, self.budget)

        # -----------------------------------------------------------------
        # 4. Initial random population.
        # -----------------------------------------------------------------
        pop = np.random.uniform(lb, ub, size=(NP, self.dim))
        fitness = np.empty(NP, dtype=float)
        for i in range(NP):
            fitness[i] = func(pop[i])
        evals = NP

        # Keep track of the best solution seen so far.
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # -----------------------------------------------------------------
        # 5. Differential Evolution main loop.
        #    DE control parameters (fixed for simplicity):
        #      F  – scale factor for the mutation difference.
        #      CR – crossover probability.
        # -----------------------------------------------------------------
        F = 0.5
        CR = 0.9

        # Run full generations while we have at least NP evaluations left.
        while evals + NP <= self.budget:
            for i in range(NP):
                # Choose three distinct indices different from i.
                candidates = list(range(NP))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Create mutant vector: mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Clip mutant to the feasible region.
                mutant = np.clip(mutant, lb, ub)

                # Binomial crossover: mix mutant with current individual.
                trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[i])

                # Ensure at least one dimension comes from the mutant.
                rnd_dim = np.random.randint(self.dim)
                trial[rnd_dim] = mutant[rnd_dim]

                # Evaluate trial candidate.
                trial_f = func(trial)
                evals += 1

                # Selection: keep the better of trial and original.
                if trial_f <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_y:
                        best_x = trial.copy()
                        best_y = trial_f

        # -----------------------------------------------------------------
        # 6. Use any remaining evaluation budget for random sampling.
        # -----------------------------------------------------------------
