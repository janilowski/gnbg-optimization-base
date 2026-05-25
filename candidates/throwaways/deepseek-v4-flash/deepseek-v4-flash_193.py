# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a standard Differential Evolution (DE/rand/1/bin) algorithm for continuous black-box minimization.
# Search state: A population of candidate solutions (vectors) and their associated fitness values.
# Candidate generation: For each individual, a mutant vector is created by adding the weighted difference between two random population members to a third. Then, binomial crossover with the parent produces a trial vector.
# Selection and replacement: Greedy selection: the trial replaces the parent if it yields lower (better) fitness.
# Adaptation: None (fixed parameters F=0.5, CR=0.9).
# Exploration mechanisms: Mutation uses random differential variations; crossover allows mixing; initial population random uniform.
# Exploitation mechanisms: Selection pressure towards better solutions; population gradually concentrates.
# Boundary handling: Clipping trial vectors to the lower and upper bounds.
# Budget strategy: Population size is set based on dimension and budget; the algorithm runs generations until the evaluation budget is exhausted. The remaining budget may result in partial generation cycles.
# Closest known influences: Classic DE (Storn & Price, 1997).
# Novelty or unusual aspects: None; straightforward implementation.
# Failure modes: May stagnate on highly multimodal or deceptive functions if population size too small; fixed parameters may not adapt to problem characteristics; clipping may cause loss of diversity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            low = np.asarray(func.lower, dtype=float)
            high = np.asarray(func.upper, dtype=float)
        else:
            lb = func.bounds.lb
            ub = func.bounds.ub
            low = np.atleast_1d(np.asarray(lb, dtype=float))
            high = np.atleast_1d(np.asarray(ub, dtype=float))

        if low.ndim == 0:
            low = low * np.ones(self.dim)
            high = high * np.ones(self.dim)

        dim = self.dim
        evals_remaining = self.budget

        # Population size: at least 10, at most 4*dim, but never exceed budget
        NP = min(evals_remaining, max(10, 4 * dim))
        evals_remaining -= NP

        # Initialize population uniformly within bounds
        pop = np.random.uniform(low, high, (NP, dim))
        fitness = np.empty(NP)
        for i in range(NP):
            fitness[i] = func(pop[i])

        # Track best so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # DE parameters
        F = 0.5
        CR = 0.9

        # Main loop: generations until budget runs out
        pop_indices = np.arange(NP)
        while evals_remaining > 0:
            # Permute the order of processing to avoid bias
            for i in np.random.permutation(pop_indices):
                if evals_remaining <= 0:
                    break

                # Choose three distinct indices different from i
                candidates = np.random.choice(pop_indices[pop_indices != i], 3, replace=False)
                a, b, c = candidates

                # Mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover: binomial
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                # Boundary handling: clip to bounds
                trial = np.clip(trial, low, high)

                # Evaluation
                trial_fitness = func(trial)
                evals_remaining -= 1

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

            # Update best if it changed, or just rely on tracking above
            # (already updated inside loop)

        return best_x, best_y
