# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a Differential Evolution (DE) algorithm for black-box minimization.
# The algorithm uses the classic DE/rand/1/bin strategy with a dithering scaling factor and binomial crossover.
# Search state: A population of candidate solutions (points) and their corresponding objective values.
# The best found solution and its value are tracked across generations.
# Candidate generation: For each target vector, a mutant vector is created by adding a scaled difference
# between two random population vectors to a third random vector (DE/rand/1). The scaling factor F is
# sampled uniformly from [0.5, 1.0] per target (dither). Then binomial crossover with fixed rate CR
# combines the mutant with the target to produce a trial vector.
# Selection and replacement: Greedy selection – the trial vector replaces the target if it yields a lower
# objective value; otherwise the target survives to the next generation.
# Adaptation: No online parameter adaptation is used; the algorithm employs fixed CR and a dithering range for F
# to maintain diversity.
# Exploration mechanisms: Random differential vectors and dither on F promote exploration.
# Exploitation mechanisms: As the population converges, exploitation increases via selection pressure.
# Boundary handling: Trial vectors that violate bounds are reflected inside using a bounce-back scheme.
# Budget strategy: Evaluation count is tracked precisely; the algorithm stops when the budget is exhausted,
# returning the best solution observed.
# Closest known influences: Classic DE/rand/1/bin (Storn & Price, 1997) with dither (Price et al., 2005).
# Novelty or unusual aspects: None – standard DE with basic features.
# Failure modes: If the population size is too small for the dimension, premature convergence may occur.
# Also, a limited budget may prevent sufficient generations.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds from func object")

        dim = self.dim
        budget = self.budget

        # Population size – adaptive to dimension and budget
        pop_size = max(10, dim)          # sensible default
        if pop_size * 2 > budget:        # at least 2 generations possible
            pop_size = max(4, budget // 2)

        # Algorithm parameters
        F_low = 0.5
        F_high = 1.0
        CR = 0.9

        # Initialize population uniformly within bounds
        pop = np.random.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)

        # Evaluate initial population
        evals = 0
        best_x = None
        best_y = np.inf
        for i in range(pop_size):
            if evals >= budget:
                break
            y = func(pop[i])
            fitness[i] = y
            evals += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Main DE loop
        generation = 0
        while evals < budget:
            new_pop = pop.copy()
            indices = list(range(pop_size))
            for i in indices:
                if evals >= budget:
                    break

                # Choose three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 3:
                    break
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Dither: scale factor per individual
                F = np.random.uniform(F_low, F_high)

                # Mutation: v = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary handling – bounce back
                for j in range(dim):
                    if trial[j] < lb[j]:
                        trial[j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                    elif trial[j] > ub[j]:
                        trial[j] = ub[j] - np.random.rand() * (ub[j] - lb[j])

                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # Selection: greedy
                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

            pop = new_pop
            generation += 1

        return best_x, best_y
