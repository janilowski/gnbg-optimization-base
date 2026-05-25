import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a Differential Evolution (DE) optimizer for
# black-box minimization. DE is a population-based stochastic search algorithm
# that maintains a set of candidate solutions and iteratively improves them
# through mutation, crossover, and selection.
#
# Search state: The state consists of a population of NP candidate vectors,
# each of length dim, stored as a 2D numpy array. The best solution found so
# far (best_x) and its objective value (best_y) are tracked.
#
# Candidate generation: For each target vector in the population, a mutant
# vector is generated using the DE/rand/1 scheme: mutant = r1 + F * (r2 - r3),
# where r1, r2, r3 are distinct random population members (different from the
# target). The mutant is then crossed with the target via binomial crossover
# (crossover probability CR) to produce a trial vector.
#
# Selection and replacement: If the trial vector has a lower (better) objective
# value than the target, it replaces the target in the next generation.
# Otherwise, the target remains.
#
# Adaptation: The population size NP is determined at initialization based on
# the evaluation budget and dimension, ensuring a reasonable number of
# generations. The control parameters F (scale factor) and CR (crossover rate)
# are fixed (F=0.8, CR=0.9). No online adaptation is used to keep the
# implementation simple and robust.
#
# Exploration mechanisms: The mutation operator (difference vector) introduces
# exploration by exploring directions between random individuals. The
# crossover operator combines parts of the target and mutant, allowing
# discovery of new areas.
#
# Exploitation mechanisms: The selection operator retains the better of target
# and trial, gradually focusing the population around promising regions. The
# best solution is tracked across all generations.
#
# Boundary handling: After mutation and crossover, trial vector components
# are clipped to the search bounds defined by the function (lower/upper or
# bounds.lb/.ub). This keeps all candidates within the feasible domain.
#
# Budget strategy: The total number of function evaluations is strictly capped
# by the provided budget. NP evaluations are used for initial population
# evaluation, and each subsequent generation uses NP evaluations for trial
# vectors. The loop stops when the remaining budget is less than NP.
#
# Closest known influences: Standard Differential Evolution (DE/rand/1/bin)
# as described in Storn & Price (1997).
#
# Novelty or unusual aspects: None; this is a straightforward implementation.
# The population size is automatically scaled with budget and dimension to
# balance exploration and exploitation under limited evaluations.
#
# Failure modes: On very small budgets (less than a small population), the
# algorithm may only evaluate the initial population without any generation
# updates, effectively performing a random search. On high-dimensional
# problems with tight budgets, performance may degrade due to insufficient
# population diversity.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget: int, dim: int):
        """Initialize the DE optimizer.

        Args:
            budget: Maximum number of function evaluations allowed.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # Determine population size: at least 4 and at most 100,
        # scaled so that we can run at least 2 generations (init + one update).
        # This is a heuristic; for very small budgets we may only do init.
        if budget < 20:
            self.NP = max(4, budget // 4)      # minimal population
        else:
            # Aim for about (budget / NP) ≈ 2*dim generations, but cap NP.
            self.NP = max(4, min(100, int(budget / (2 * dim + 1))))

        # Standard DE control parameters
        self.F = 0.8   # mutation scale factor
        self.CR = 0.9  # crossover probability

        # The best solution and its value will be stored during runs
        self.best_x = None
        self.best_y = np.inf

    def __call__(self, func):
        """Run the DE optimization.

        Args:
            func: The objective function. Must provide either
                  func.lower, func.upper or func.bounds.lb, func.bounds.ub.
                  The function is called with a 1-D array and returns a float.

        Returns:
            (best_x, best_y): The best found solution and its objective value.
        """
        # ---------- Read bounds ----------
        lower = getattr(func, 'lower', None)
        upper = getattr(func, 'upper', None)
        if lower is None or upper is None:
            # try bounds attribute
            bounds = getattr(func, 'bounds', None)
            if bounds is not None:
                lower = getattr(bounds, 'lb', None)
                upper = getattr(bounds, 'ub', None)
        if lower is None or upper is None:
            raise ValueError("Function must provide bounds via .lower/.upper "
                             "or .bounds.lb/.bounds.ub")
        # Ensure arrays
        lb = np.asarray(lower, dtype=float).reshape(1, -1)
        ub = np.asarray(upper, dtype=float).reshape(1, -1)
        dim = self.dim

        # ---------- Initialize population ----------
        NP = self.NP
        # Uniform random within bounds
        population = lb + (ub - lb) * np.random.rand(NP, dim)
        fitness = np.empty(NP)
        evals = 0

        # Evaluate initial population
        for i in range(NP):
            fitness[i] = func(population[i])
            evals += 1
            if fitness[i] < self.best_y:
                self.best_y = fitness[i]
                self.best_x = population[i].copy()

        # ---------- Main DE loop ----------
        # Each generation uses NP evaluations (trial vectors).
        # Stop when remaining budget < NP (cannot complete a generation).
        while evals + NP <= self.budget:
            for i in range(NP):
                # Mutation: pick three distinct random indices != i
                r = np.random.choice([j for j in range(NP) if j != i], 3,
                                     replace=False)
                r1, r2, r3 = r
                mutant = population[r1] + self.F * (population[r2] - population[r3])
                # Crossover: binomial
                cross_mask = np.random.rand(dim) < self.CR
                # Ensure at least one dimension is inherited from mutant
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(dim)] = True
                trial = np.where(cross_mask, mutant, population[i])
                # Boundary handling: clip to bounds
                trial = np.clip(trial, lb, ub).flatten()
                # Evaluate trial
                trial_fit = func(trial)
                evals += 1
                # Selection
                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < self.best_y:
                        self.best_y = trial_fit
                        self.best_x = trial.copy()
            # Check budget after finishing a generation (already checked loop condition)

        # Return best found
        return self.best_x, self.best_y
