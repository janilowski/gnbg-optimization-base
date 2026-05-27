# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a classic Differential Evolution (DE) algorithm for black-box minimization.
# Search state: A population of candidate solutions (real-valued vectors) maintained as a 2D numpy array,
#               along with their corresponding objective values in a 1D array.
# Candidate generation: For each target vector, a mutant vector is created by adding the scaled difference
#                       between two randomly selected population members to a third base vector (DE/rand/1).
#                       Then binomial crossover combines the mutant with the target to form a trial vector.
# Selection and replacement: Greedy selection: if the trial vector has a lower (better) objective value
#                            than the target, it replaces the target in the population.
# Adaptation: No online adaptation; parameters F (scaling factor) and CR (crossover rate) are fixed.
# Exploration mechanisms: Mutation via random differences introduces diversity; crossover can combine
#                         information from different solutions.
# Exploitation mechanisms: Greedy selection retains improving solutions; population gradually converges
#                          around promising regions.
# Boundary handling: Trial vectors are clipped component-wise to the variable bounds [lower, upper].
# Budget strategy: The population size is set to min(10 * dim, budget // 2) to ensure at least one full
#                  generation. The number of generations is floor((budget - population_size) / population_size)
#                  so total evaluations never exceed budget. Initial population is evaluated separately.
# Closest known influences: Storn & Price (1997) – Differential Evolution.
# Novelty or unusual aspects: None; this is a standard, well-known implementation with fixed parameters.
# Failure modes: On highly multimodal or deceptive landscapes, the fixed parameters may cause premature
#                convergence. Very high dimensions may require larger populations than the budget allows.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the Differential Evolution optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # DE parameters (fixed)
        self.F = 0.8       # scaling factor
        self.CR = 0.9      # crossover probability

        # Population size: scale with dimension, but never exceed budget/2 to allow at least one generation
        self.popsize = min(10 * dim, budget // 2)
        # Ensure at least 4 individuals (needed for mutation)
        if self.popsize < 4:
            self.popsize = min(budget, 4)
        if self.popsize < 4:
            self.popsize = budget  # fallback: just random sampling

        # Number of generations (after initial evaluation)
        self.max_gens = (budget - self.popsize) // self.popsize

    def __call__(self, func):
        """
        Run the DE optimizer on the given objective function.

        Parameters
        ----------
        func : callable
            Objective function to minimize. Must have attributes lower/upper or bounds.lb/bounds.ub.

        Returns
        -------
        best_x : np.ndarray
            Best found solution.
        best_y : float
            Objective value at best_x.
        """
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lower = np.asarray(b.lb, dtype=float)
            upper = np.asarray(b.ub, dtype=float)
        else:
            raise AttributeError("Objective function must provide lower/upper or bounds.lb/bounds.ub")

        # Broadcast to dimension size
        if lower.ndim == 0:
            lower = np.full(self.dim, lower)
        if upper.ndim == 0:
            upper = np.full(self.dim, upper)

        # Initialize population uniformly in bounds
        rng = np.random.default_rng()  # harness sets seed globally
        pop = rng.uniform(lower, upper, size=(self.popsize, self.dim))

        # Evaluate initial population
        evals = 0
        fitness = np.empty(self.popsize)
        for i in range(self.popsize):
            fitness[i] = func(pop[i])
            evals += 1

        # Track best so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        for gen in range(self.max_gens):
            if evals >= self.budget:
                break

            # For each target vector
            for i in range(self.popsize):
                if evals >= self.budget:
                    break

                # Choose three distinct random indices different from i
                idxs = [idx for idx in range(self.popsize) if idx != i]
                r1, r2, r3 = rng.choice(idxs, size=3, replace=False)

                # Mutation: DE/rand/1
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])

                # Binomial crossover
                # Generate a random dimension index to ensure at least one crossover
                j_rand = rng.integers(self.dim)
                trial = np.empty(self.dim)
                for j in range(self.dim):
                    if rng.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i, j]

                # Boundary handling: clip to bounds
                trial = np.clip(trial, lower, upper)

                # Evaluate trial
                trial_fit = func(trial)
                evals += 1

                # Greedy selection
                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

        return best_x, best_y
