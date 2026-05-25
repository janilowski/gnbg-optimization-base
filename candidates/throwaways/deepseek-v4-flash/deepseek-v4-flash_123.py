import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact Differential Evolution (DE) algorithm,
# using rand/1/bin, for minimizing a black-box objective function within a fixed
# evaluation budget.  It is designed to be robust across a wide range of dimensions
# and budget sizes by falling back to random search when the budget is too small.
# Search state: A population of candidate vectors stored as a 2D numpy array (pop_size x dim).
# Also keeps the current best solution and its fitness.
# Candidate generation: For each target vector, three distinct random vectors are
# chosen from the population.  A mutant vector is computed as the weighted
# difference of two of them added to the third.  Then a trial vector is formed
# by binomial crossover between the mutant and target, with fixed crossover rate CR.
# Selection and replacement: Greedy one-to-one: the trial replaces the target if
# its fitness is not worse (minimization).  The global best is updated whenever a
# better fitness is found.
# Adaptation: No dynamic parameter adaptation; F (scale factor) and CR are fixed
# at 0.8 and 0.9 respectively.
# Exploration mechanisms: The mutation operator uses random difference vectors,
# which promotes exploration by generating diverse trial points.  The population
# maintains diversity through the greedy selection that only replaces inferior
# individuals.
# Exploitation mechanisms: Crossover retains components from the target vector,
# preserving good building blocks.  The global best is also preserved and used
# only for output, not for generation.
# Boundary handling: Components that fall outside the search space are clipped to
# the closest bound.
# Budget strategy: The algorithm stops exactly when the remaining budget is
# exhausted; a separate counter tracks all calls to the objective function.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997)
# with no additional adaptation.
# Novelty or unusual aspects: A simple fallback to random search for extremely low
# budgets (below 4 evaluations) ensures the algorithm is always executable.
# Failure modes: With a population that is too small for the problem’s modality,
# the algorithm may converge prematurely.  Fixed parameters may be suboptimal
# on some landscapes.  The algorithm does not specifically handle noisy
# objectives.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    """Differential Evolution minimizer with fixed parameters.

    Parameters
    ----------
    budget : int
        Maximum number of function evaluations allowed.
    dim : int
        Dimensionality of the search space.
    """
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """Run the optimization.

        Parameters
        ----------
        func : callable
            Objective function to minimize.  Must expose bounds either as
            `func.lower` / `func.upper` or as `func.bounds.lb` / `func.bounds.ub`.

        Returns
        -------
        best_x : numpy.ndarray, shape (dim,)
            Best solution found.
        best_y : float
            Corresponding function value.
        """
        # --- Determine bounds -------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via func.lower/upper "
                                 "or func.bounds.lb/ub")

        dim = self.dim
        budget = self.budget

        # --- Handle very small budget: switch to random search ---------------
        if budget < 4:
            best_y = float('inf')
            best_x = np.empty(dim)
            for _ in range(budget):
                x = lb + np.random.rand(dim) * (ub - lb)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # --- Set population size ----------------------------------------------
        # Heuristic: base size 30, scale linearly with dimension, but never use
        # more than half the budget (so at least one generation is possible).
        pop_size = max(4 * dim, 30)
        pop_size = min(pop_size, budget // 2)
        # Ensure at least 4 individuals (needed for DE mutation)
        pop_size = max(pop_size, 4)

        # --- Parameters -------------------------------------------------------
        F = 0.8      # mutation factor
        CR = 0.9     # crossover rate

        # --- Initialisation ---------------------------------------------------
        # Uniform random population within bounds
        pop = lb + np.random.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1

        # Track global best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # --- Main DE loop -----------------------------------------------------
        while evals < budget:
            # Stop if we cannot complete even one trial per individual
            # (we need pop_size evaluations to complete one generation)
            if evals + pop_size > budget:
                break

            # For each target, generate a trial vector
            for i in range(pop_size):
                # Choose three distinct random indices different from i
                indices = [j for j in range(pop_size) if j != i]
                r = np.random.choice(indices, 3, replace=False)
                a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]

                # Mutation: v = a + F * (b - c)
                mutant = a + F * (b - c)

                # Binomial crossover with target pop[i]
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                # Ensure trial inside bounds (clip)
                trial = np.clip(trial, lb, ub)

                # Evaluate
                y_trial = func(trial)
                evals += 1

                # One-to-one greedy selection
                if y_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = y_trial
                    # Update global best if needed
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()

            # After each generation, check if we can still go on
            # (the while condition will also break after the next iteration)

        # If any budget remains after the loop, spend it by random perturbations
        # around the current best (local refinement)
        while evals < budget:
            x = best_x + 0.1 * (ub - lb) * np.random.randn(dim)
            x = np.clip(x, lb, ub)
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()

        return best_x, best_y
