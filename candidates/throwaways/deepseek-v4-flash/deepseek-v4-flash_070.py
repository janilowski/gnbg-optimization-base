import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a Differential Evolution (DE) algorithm with rand/1/bin mutation and binomial crossover for black-box minimization. It is designed for the GNBG benchmark.
# Search state: Population of candidate solutions with associated fitness values, plus best-so-far solution.
# Candidate generation: For each parent, a mutant vector is created by adding the scaled difference of two other random population members to a third random member (rand/1). Then binomial crossover with the parent yields a trial vector.
# Selection and replacement: Greedy selection: if the trial vector's fitness is better (lower) than the parent's, it replaces the parent; otherwise, the parent remains.
# Adaptation: None; fixed differential weight F=0.5 and crossover rate CR=0.9. No parameter adaptation.
# Exploration mechanisms: The difference vector provides random perturbation; the random base vector in mutation and independent crossover for each dimension maintain diversity.
# Exploitation mechanisms: The crossover with the parent ensures that good components are preserved, and selection pushes the population toward better regions.
# Boundary handling: Reflective boundary handling: components outside bounds are reflected back into the feasible domain.
# Budget strategy: The algorithm stops immediately after the evaluation budget is exhausted; no extra evaluations are performed.
# Closest known influences: Classic DE/rand/1/bin as described by Storn and Price (1997).
# Novelty or unusual aspects: None; a straightforward implementation with no adaptation.
# Failure modes: May converge prematurely on multimodal functions due to lack of diversity preservation; fixed parameters may not suit all problems; performance may degrade with high dimensions relative to population size.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    """
    Differential Evolution (DE/rand/1/bin) for minimization.
    """

    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def _reflect(self, x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Reflect out-of-bounds components back into [lb, ub]."""
        x = np.where(x < lb, 2 * lb - x, x)
        x = np.where(x > ub, 2 * ub - x, x)
        # Clamp in case of multiple reflections (e.g., very far out)
        x = np.clip(x, lb, ub)
        return x

    def __call__(self, func):
        """
        Run DE optimization.

        Parameters
        ----------
        func : callable
            Objective function to minimize. Must be a callable that accepts a 1-D array
            and returns a scalar. The function object must provide lower and upper bounds
            via either `func.lower` / `func.upper` or `func.bounds.lb` / `func.bounds.ub`.

        Returns
        -------
        best_x : np.ndarray
            Best solution found.
        best_y : float
            Best objective value found.
        """
        # ----- Read bounds -------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)
        else:
            raise AttributeError("Objective must provide bounds via lower/upper or bounds.lb/ub")

        # Ensure bounds are 1-D arrays of the right dimension
        lb = np.broadcast_to(lb, (self.dim,)).copy()
        ub = np.broadcast_to(ub, (self.dim,)).copy()

        # ----- Algorithm parameters ----------------------------------------
        # Population size: roughly 10*dim but not more than budget/2, and at least 4 (for DE mutation)
        pop_size = min(10 * self.dim, self.budget // 2)
        pop_size = max(pop_size, 4)
        # Differential weight and crossover rate
        F = 0.5
        CR = 0.9

        # ----- Initialization ---------------------------------------------
        # Uniformly random population within bounds
        population = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        fitness = np.full(pop_size, np.inf)
        # Evaluate all initial individuals
        evaluations = 0
        for i in range(pop_size):
            fitness[i] = func(population[i])
            evaluations += 1
            if evaluations >= self.budget:
                # Budget exhausted during initialization – return best so far
                best_idx = np.argmin(fitness[:evaluations])
                return population[best_idx].copy(), fitness[best_idx]

        # Track best solution
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]

        # ----- Main loop --------------------------------------------------
        while evaluations < self.budget:
            # For each individual, generate a trial vector
            for i in range(pop_size):
                if evaluations >= self.budget:
                    break

                # Choose three distinct random indices different from i
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, size=3, replace=False)

                # Mutant vector: base + F * (difference)
                mutant = population[a] + F * (population[b] - population[c])

                # Binomial crossover with parent
                trial = population[i].copy()
                # Select at least one dimension to cross (ensure change)
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary handling (reflection)
                trial = self._reflect(trial, lb, ub)

                # Evaluate trial
                trial_fitness = func(trial)
                evaluations += 1

                # Greedy selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    # Update global best
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
