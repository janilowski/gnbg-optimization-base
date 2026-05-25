# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a simple Differential Evolution (DE) optimizer for black‑box minimization. DE maintains a population of candidate solutions, creates new candidates by mutating three random individuals, recombines the mutant with the target using binomial crossover, and greedily selects the better of the two. The population size is automatically chosen based on the problem dimension and available evaluation budget, with a fallback to pure random search when the budget is too small to form a viable DE population.
# Search state: Population array (NP × dim) of candidates and their objective values. Additionally the best observed solution (best_x, best_y) is tracked throughout the search.
# Candidate generation: For each target vector, three distinct random indices are selected to form a mutant vector via `mutant = pop[a] + F * (pop[b] - pop[c])` with a fixed scaling factor F = 0.8. Binomial crossover with probability CR = 0.9 generates a trial vector; at least one dimension is always taken from the mutant to guarantee exploration.
# Selection and replacement: Greedy (μ‑like) selection – the trial replaces the target only if its objective value is not larger.
# Adaptation: No explicit adaptation of DE control parameters; F and CR are fixed at 0.8 and 0.9, respectively.
# Exploration mechanisms: Mutation introduces diversity; random sampling is used to consume any remaining evaluations when the remaining budget is insufficient for a full generation.
# Exploitation mechanisms: Greedy selection ensures that each generation keeps the best individuals, steadily improving the current best solution.
# Boundary handling: All vectors are clipped to the user‑supplied bounds after mutation and during initialization, preventing evaluation outside the feasible region.
# Budget strategy: The algorithm never exceeds the supplied evaluation budget. If the budget is smaller than the chosen population size, a pure random search is performed. When the final generation cannot be completed, leftover evaluations are spent on random points.
# Closest known influences: Storn & Price’s canonical Differential Evolution (DE/rand/1/bin).
# Novelty or unusual aspects: Automatic population sizing based on dimension and budget, plus a built‑in fallback to random search for very small budgets, provides robustness across diverse benchmark scenarios.
# Failure modes: On highly multi‑modal landscapes with extremely limited budget, DE may converge prematurely. When dimensionality is large and the budget is modest, the population may be too small to cover the space adequately, leading to sub‑optimal solutions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple Differential Evolution (DE) optimizer for black‑box minimization.

    The class conforms to the required interface:
        - __init__(self, budget, dim)
        - __call__(self, func) -> (best_x, best_y)

    Parameters
    ----------
    budget : int
        Maximum number of function evaluations allowed.
    dim : int
        Dimensionality of the search space.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        """
        Run the optimizer on the given function.

        Parameters
        ----------
        func : callable
            A black‑box function that accepts a 1‑D numpy array of length dim
            and returns a scalar objective value.

        Returns
        -------
        best_x : numpy.ndarray
            The solution vector that achieved the lowest observed objective.
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------
        # Determine problem bounds
        # ------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # Assume bounds is an object with .lb and .ub attributes
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback to a default box if no bounds are supplied
            lower = np.full(self.dim, -5.0, dtype=float)
            upper = np.full(self.dim, 5.0, dtype=float)

        if lower.shape != (self.dim,):
            lower = np.full(self.dim, -5.0, dtype=float)
        if upper.shape != (self.dim,):
            upper = np.full(self.dim, 5.0, dtype=float)

        # Ensure lower < upper (sanity check)
        if np.any(lower >= upper):
            lower = np.full(self.dim, -5.0, dtype=float)
            upper = np.full(self.dim, 5.0, dtype=float)

        # ------------------------------------------------------------
        # Configure DE parameters
        # ------------------------------------------------------------
        # Population size scales with dimension and budget, but we need at least 2 individuals.
        NP = max(2, min(10 * self.dim, self.budget // 2))
        if self.budget <= NP:
            # Not enough evaluations to form a meaningful DE population – use random search.
            return self._random_search(func, lower, upper)

        # Initialise population uniformly inside the bounds
        pop = np.random.uniform(lower, upper, size=(NP, self.dim))

        # Evaluate initial population
        fitness = np.array([func(pop[i]) for i in range(NP)], dtype=float)
        evals = NP

        # Track the best solution found so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Fixed DE control parameters
        F = 0.8   # mutation scaling factor
        CR = 0.9  # crossover probability

        # ------------------------------------------------------------
        # Main DE loop
        # ------------------------------------------------------------
        while evals < self.budget:
            remaining = self.budget - evals

            # If we cannot afford a full next generation, spend remaining budget on random points.
            if remaining < NP:
                extra_points = np.random.uniform(lower, upper, size=(remaining, self.dim))
                for xi in extra_points:
                    yi = func(xi)
                    evals += 1
                    if yi < best_y:
                        best_y = yi
                        best_x = xi.copy()
                break

            # Create new generation containers
            new_pop = np.empty_like(pop)
            new_fitness = np.empty(NP, dtype=float)

            for i in range(NP):
                # Select three distinct indices for mutation, different from i
                indices = list(range(NP))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Generate mutant vector
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lower, upper)

                # Binomial crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]   # ensure at least one component from mutant

                # Evaluate trial candidate
                y_trial = func(trial)
                evals += 1

                # Greedy selection
                if y_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = y_trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]

            # Replace old population with the new one
            pop = new_pop
            fitness = new_fitness

        return best_x, best_y

    def _random_search(self, func, lower, upper):
        """
        Fallback pure random search when the budget is too small to form a DE population.
        """
        n = self.budget
        points = np.random.uniform(lower, upper, size=(n, self.dim))
        best_x = None
        best_y = np.inf
        for pt in points:
            y = func(pt)
            if y < best_y:
                best_y = y
                best_x = pt.copy()
        # In the degenerate case n == 0, return a random feasible point
        if best_x is None:
            best_x = np.random.uniform(lower, upper, size=self.dim)
            best_y = func(best_x)
        return best_x, best_y
