# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a variant of Differential Evolution (DE/rand/1/bin) for
#     black-box minimization. The algorithm is compact, robust, and scales to
#     any dimension. It uses a fixed population size and standard mutation and
#     crossover operators with simple clipping for boundary handling.
# Search state: A population of candidate solutions (numpy array of shape
#     (pop_size, dim)) and their corresponding objective values (numpy array of
#     shape (pop_size,)). The best-known solution and its value are stored separately.
# Candidate generation: For each population member, a trial vector is created
#     by DE mutation (rand/1): base = random individual, difference = two other
#     distinct random individuals, scaled by F. Then binomial crossover with the
#     target vector using probability CR.
# Selection and replacement: After evaluation, the trial replaces the target if
#     its objective value is smaller (minimization). Greedy selection.
# Adaptation: No adaptation of control parameters (F and CR) during the run.
#     Fixed population size based on dimension: pop_size = max(10, 4 * dim).
# Exploration mechanisms: The random selection of base and difference vectors
#     provides global exploration. The fixed F and CR maintain variability.
# Exploitation mechanisms: Crossover and mutation exploit the population's
#     current distribution; selection pressure retains better solutions.
# Boundary handling: Trial vectors that violate bounds are clipped (trimmed)
#     component-wise to the feasible range [lb, ub].
# Budget strategy: The number of function evaluations is strictly limited by
#     the given budget. The algorithm stops immediately when the budget is
#     exhausted, even mid-generation.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997)
#     with rand/1 mutation and binomial crossover.
# Novelty or unusual aspects: None; the implementation is a straightforward
#     baseline DE. It prioritizes simplicity and robustness over peak performance.
# Failure modes: May converge prematurely on highly multimodal or deceptive
#     landscapes due to lack of adaptation. Population size and parameters are
#     fixed and may be suboptimal for certain problem dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initialize the Differential Evolution algorithm.

        Parameters
        ----------
        budget : int
            Total number of allowed function evaluations.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # Control parameters (common defaults)
        self.F = 0.8          # mutation factor
        self.CR = 0.9         # crossover probability

        # Population size: scale with dimension, minimum 10
        self.pop_size = max(10, 4 * dim)

    def __call__(self, func):
        """
        Run the optimizer on the given black-box function.

        Parameters
        ----------
        func : callable
            Objective function to minimize. Must have attributes `lower` and
            `upper` (or `bounds.lb` and `bounds.ub`) providing the search bounds.

        Returns
        -------
        best_x : np.ndarray
            Best found point.
        best_y : float
            Best found objective value.
        """
        # Read bounds (handle both attribute naming conventions)
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget

        # Initialize population uniformly in the search space
        population = lb + np.random.rand(pop_size, dim) * (ub - lb)

        # Evaluate initial population
        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(population[i])
            evals += 1
            if evals >= budget:
                break

        # Track best solution
        best_idx = np.argmin(fitness[:evals])
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        generation = 0
        while evals < budget:
            # For each target vector, generate a trial vector
            for i in range(pop_size):
                if evals >= budget:
                    break

                # Select three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutation: DE/rand/1
                mutant = population[r1] + self.F * (population[r2] - population[r3])

                # Crossover: binomial
                # For each dimension, randomly choose from mutant or target
                cross_points = np.random.rand(dim) < self.CR
                # Ensure at least one dimension is taken from the mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Boundary handling: clip to [lb, ub]
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # Selection: greedy (minimization)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

            generation += 1

        return best_x, best_y
