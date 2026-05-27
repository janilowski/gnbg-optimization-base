import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a Differential Evolution (DE) optimizer for
# black-box minimization. It uses the classic DE/rand/1/bin strategy with fixed
# parameters F=0.5 and CR=0.9. The population size is automatically chosen based
# on the dimension and the available budget.
#
# Search state: A population of candidate solutions (vectors) and their
# corresponding objective values. The population is stored as a NumPy array of
# shape (NP, dim).
#
# Candidate generation: For each parent in the population, a mutant vector is
# created by adding the scaled difference of two other randomly selected
# population members to a third (base) member. Then binomial crossover mixes the
# mutant with the parent to produce a trial vector.
#
# Selection and replacement: Greedy selection: if the trial vector has a lower
# objective value than the parent, it replaces the parent in the population.
#
# Adaptation: There is no parameter adaptation in this simple implementation;
# F and CR are fixed at common default values (0.5 and 0.9). The population
# size is chosen adaptively based on dimension and budget, but stays constant
# during the run.
#
# Exploration mechanisms: The differential mutation operator can generate
# large exploratory steps when the population is spread out. Crossover
# recombines information from different individuals.
#
# Exploitation mechanisms: As the population converges, the difference vectors
# become small, leading to finer local search. Greedy selection ensures that
# only improving moves are accepted.
#
# Boundary handling: Trial vectors are clipped to the variable bounds. This
# is a simple and robust approach that keeps the population inside the feasible
# region.
#
# Budget strategy: The budget is consumed by evaluating the initial population
# and then one trial per individual per generation. If the remaining budget
# cannot accommodate a full generation, only the necessary number of trial
# evaluations are performed. This guarantees that the total number of
# evaluations never exceeds the given budget.
#
# Closest known influences: The algorithm is a textbook implementation of
# Differential Evolution (Storn & Price, 1997). It follows the classic
# rand/1/bin scheme.
#
# Novelty or unusual aspects: Nothing novel; the implementation is a
# straightforward, compact variant intended for robustness and clarity.
# An adaptive population size is used to accommodate small budgets.
#
# Failure modes: On highly multimodal landscapes, DE may converge prematurely
# to local optima if the population loses diversity. The fixed parameters
# may not be optimal for all problems. The clipping boundary handling can
# lead to loss of diversity when many trial vectors are stuck at the bounds.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # Fixed parameters for DE/rand/1/bin
        self.F = 0.5
        self.CR = 0.9
        # Choose population size based on dimension, but limit to at most budget
        # (ensuring at least 4 individuals for the DE mutation to work)
        self.NP = min(budget, max(4, 4 * dim))
        # If budget is too small to have a population of 4, reduce it to budget
        if self.NP < 4:
            self.NP = budget

    def __call__(self, func):
        # Obtain bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # Assume bounds is an object with attributes lb, ub
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot determine bounds from func object")

        dim = self.dim
        NP = self.NP
        F = self.F
        CR = self.CR

        # Handle the pathological case where budget is 0 (should not happen)
        if self.budget <= 0:
            return np.full(dim, np.nan), np.inf

        # Initialize population uniformly within bounds
        population = lower + (upper - lower) * np.random.rand(NP, dim)
        # Evaluate initial population
        evals = 0
        fitness = np.full(NP, np.inf)
        for i in range(NP):
            fitness[i] = func(population[i])
            evals += 1
            if evals >= self.budget:
                break

        # Find best so far
        best_idx = np.argmin(fitness[:evals])
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        while evals < self.budget:
            # Determine how many trial evaluations we can still afford this generation
            trials_remaining = self.budget - evals
            if trials_remaining <= 0:
                break
            n_trials = min(NP, trials_remaining)

            # Shuffle indices to process in random order (optional, helps diversity)
            indices = np.random.permutation(NP)

            for idx in indices[:n_trials]:
                # Select three distinct random population members different from idx
                candidates = [i for i in range(NP) if i != idx]
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                # Mutation: v = a + F * (b - c)
                mutant = population[a] + F * (population[b] - population[c])

                # Binomial crossover
                cross_points = np.random.rand(dim) <= CR
                # Ensure at least one dimension from the mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, population[idx])

                # Clip to bounds
                trial = np.clip(trial, lower, upper)

                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # Selection: keep the better vector
                if trial_fitness < fitness[idx]:
                    population[idx] = trial
                    fitness[idx] = trial_fitness
                    if trial_fitness < best_y:
                        best_x = trial.copy()
                        best_y = trial_fitness

                # Early exit if budget exhausted
                if evals >= self.budget:
                    break

        return best_x, best_y
