# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a minimal Differential Evolution (DE) optimizer for black‑box minimization. It maintains a population of candidate solutions, creates new candidates via mutation and crossover, selects the better ones, and strictly respects the evaluation budget.
# Search state: The algorithm keeps a population of NP vectors and their fitness values, plus the best solution observed so far. No state is retained between separate calls to __call__; each invocation starts fresh.
# Candidate generation: For each target vector, a mutant is built by picking three distinct individuals and adding a scaled difference between two of them (F·(b−c)). The mutant is clipped to the problem bounds.
# Selection and replacement: After crossover, the trial vector is evaluated and compared to the target vector. If the trial’s fitness is no worse, it replaces the target in the population.
# Adaptation: The DE control parameters (scaling factor F and crossover rate CR) are fixed (0.5 and 0.9) and not adapted during search. The population size scales with dimensionality to ensure sufficient diversity.
# Exploration mechanisms: Mutation with random scaling provides exploration; the high crossover rate promotes mixing of dimensions. Bounds clipping prevents leaving the feasible region.
# Exploitation mechanisms: Selection preserves the best individuals for the next generation, focusing the search on promising regions.
# Boundary handling: Any component that falls outside the user‑provided bounds is clipped to the nearest bound.
# Budget strategy: The algorithm counts function evaluations and halts immediately when the budget is exhausted. The initial population size is capped by the budget to avoid overspending.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997) using the "DE/rand/1/bin" scheme.
# Novelty or unusual aspects: The implementation is deliberately compact and relies on a fixed population size that grows linearly with dimension. No archiving or specialized adaptation mechanisms are used.
# Failure modes: If the budget is very small relative to the dimension, the search may not converge. With highly multi‑modal landscapes, insufficient population size can cause premature convergence.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple Differential Evolution (DE) optimizer for black‑box minimization.

    The class conforms to the required interface:
        - __init__(self, budget, dim)
        - __call__(self, func) -> (best_x, best_y)
    """

    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the problem.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the DE optimizer on the given black‑box function.

        Parameters
        ----------
        func : callable
            Objective function to be minimized. It can be accessed for bounds:
                - func.lower / func.upper, or
                - func.bounds.lb / func.bounds.ub

        Returns
        -------
        best_x : np.ndarray
            Best (lowest‑fitness) solution found.
        best_y : float
            Fitness value of the best solution.
        """
        # Detect problem bounds
        lower, upper = self._get_bounds(func)

        # Early exit for zero budget
        if self.budget <= 0:
            # No evaluations possible; return a dummy solution
            best_x = np.zeros(self.dim)
            best_y = np.inf
            return best_x, best_y

        # Determine population size (NP). It must be at least 4 for DE to work,
        # but we also cap it by the available budget.
        np_pop = max(10, 5 * self.dim)  # default rule of thumb
        np_pop = max(4, min(self.budget, np_pop))

        # If budget is too small to support a viable DE population, fall back to
        # simple random sampling.
        if np_pop < 4:
            return self._random_search(func, lower, upper, self.budget)

        # Initialize population uniformly within the bounds
        pop = lower + (upper - lower) * np.random.rand(np_pop, self.dim)

        # Evaluate initial population
        fitness = np.empty(np_pop)
        for i in range(np_pop):
            fitness[i] = func(pop[i])
        evals = np_pop

        # Keep track of the best solution seen so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # DE control parameters (fixed)
        F = 0.5   # scaling factor
        CR = 0.9  # crossover probability

        # Main DE loop – run until the budget is exhausted
        while evals < self.budget:
            for i in range(np_pop):
                if evals >= self.budget:
                    break

                # Select three distinct indices different from i
                others = list(range(i)) + list(range(i + 1, np_pop))
                a, b, c = np.random.choice(others, 3, replace=False)

                # Create mutant vector: x_a + F * (x_b - x_c)
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Clip mutant to the feasible region
                mutant = np.clip(mutant, lower, upper)

                # Perform crossover with the target vector
                # Generate a binary mask where each entry is taken from the mutant with probability CR
                mask = np.random.rand(self.dim) < CR
                # Ensure at least one dimension comes from the mutant
                if not np.any(mask):
                    mask[np.random.randint(self.dim)] = True

                trial = np.where(mask, mutant, pop[i])

                # Evaluate the trial candidate
                trial_fitness = func(trial)
                evals += 1

                # Selection: replace target if trial is not worse
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    # Update global best if needed
                    if trial_fitness < best_y:
                        best_x = pop[i].copy()
                        best_y = trial_fitness

        return best_x, best_y

    def _get_bounds(self, func):
        """
        Attempt to retrieve the lower and upper bounds of the search space.

        Supports two common conventions:
            - func.lower / func.upper
            - func.bounds.lb / func.bounds.ub

        If bounds cannot be found, defaults to [0, 1] per dimension.
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # Assume func.bounds has .lb and .ub attributes
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: no bounds provided – use [0, 1] per dimension
            lower = np.zeros(self.dim)
            upper = np.ones(self.dim)

        # Ensure lower/upper are NumPy arrays of correct shape
        lower = np.atleast_1d(lower)
        upper = np.atleast_1d(upper)
        if lower.shape[0] != self.dim:
            lower = np.full(self.dim, lower[0] if lower.size == 1 else 0.0)
        if upper.shape[0] != self.dim:
            upper = np.full(self.dim, upper[0] if upper.size == 1 else 1.0)

        return lower, upper

    def _random_search(self, func, lower, upper, budget):
        """
        Fallback random search when the population size would be too small for DE.

        Simply samples `budget` points uniformly within the bounds and returns the best.
        """
        best_x = None
        best_y = np.inf
        for _ in range(budget):
            x = lower + (upper - lower) * np.random.rand(self.dim)
            y = func(x)
            if y < best_y:
                best_y = y
                best_x = x.copy()
        # In the extremely unlikely case of no evaluation (budget==0), return dummy
        if best_x is None:
            best_x = lower.copy()
        return best_x, best_y
