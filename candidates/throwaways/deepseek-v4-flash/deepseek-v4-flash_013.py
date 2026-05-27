import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) implementation for black-box minimization.
#          Uses DE/rand/1/bin with per-individual dithering and static crossover rate.
# Search state: Maintains population of candidate solutions (NP vectors) and their objective values.
# Candidate generation: For each target vector, a mutant is created by adding a scaled difference
#                       between two random population members to a third random member (rand/1).
#                       Crossover (binomial) combines mutant with target to form trial vector.
# Selection and replacement: Greedy selection: if trial is better (lower f), it replaces the target.
# Adaptation: Mutation scaling factor F is randomly drawn per individual from U[0.4,0.6] (dither).
#             Crossover rate CR is fixed at 0.9.
# Exploration mechanisms: Random mutation base and difference vectors; dither provides varying step
#                         sizes; population diversity maintained by crossover.
# Exploitation mechanisms: Greedy selection preserves best solutions; difference vectors shrink as
#                          population converges, reducing step sizes.
# Boundary handling: Trial components outside bounds are clipped to the nearest bound.
# Budget strategy: Population size NP = max(4, dim) or floor(budget/2) if budget is very small.
#                  Algorithm stops when evaluation count reaches budget.
# Closest known influences: Classic DE (Storn & Price, 1997) with rand/1/bin strategy.
# Novelty or unusual aspects: Extremely minimal implementation, no adaptation beyond simple dither.
#                             Designed for robustness across dimensions and tight budgets.
# Failure modes: May converge prematurely on multimodal landscapes due to lack of restart or
#                diversity preservation mechanisms. Static CR may be suboptimal for certain problems.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        budget = self.budget
        dim = self.dim

        # Determine bounds from func (handle both attribute styles)
        try:
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)

        # Ensure bounds are 1D arrays
        if lower.ndim == 0:
            lower = np.full(dim, lower)
            upper = np.full(dim, upper)

        # Population size: at least 4 and at least dim, but limited by budget
        min_pop = max(4, dim)
        np_size = min(min_pop, budget // 2)  # at most half budget for initial eval
        if np_size < 4:
            np_size = max(2, min_pop)  # ensure at least 2 for mutation

        # Parameter settings
        cr = 0.9          # crossover rate
        f_low = 0.4       # lower bound for dither
        f_high = 0.6      # upper bound

        # Initialize population uniformly in bounds
        pop = np.random.uniform(lower, upper, size=(np_size, dim))
        evals = 0

        # Evaluate initial population
        fitness = np.empty(np_size)
        for i in range(np_size):
            fitness[i] = func(pop[i])
            evals += 1
            if evals >= budget:
                break

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        while evals < budget:
            # If population is too small for mutation, fall back to random search
            if np_size < 4:
                x = np.random.uniform(lower, upper)
                y = func(x)
                evals += 1
                if y < best_y:
                    best_x = x
                    best_y = y
                continue

            # For each individual, generate trial vector
            for i in range(np_size):
                if evals >= budget:
                    break

                # Choose three distinct random indices different from i
                candidates = list(range(np_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    # Not enough candidates; skip mutation
                    break
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # Mutation (rand/1) with dither: F chosen per individual
                f = np.random.uniform(f_low, f_high)
                mutant = pop[r1] + f * (pop[r2] - pop[r3])

                # Binomial crossover
                j_rand = np.random.randint(0, dim)
                trial = np.where(np.random.rand(dim) < cr, mutant, pop[i])
                # Always take at least one component from mutant
                trial[j_rand] = mutant[j_rand]

                # Bounce-back: clip to bounds
                trial = np.clip(trial, lower, upper)

                # Evaluate trial
                trial_y = func(trial)
                evals += 1

                # Selection: greedy
                if trial_y < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_y
                    if trial_y < best_y:
                        best_x = trial.copy()
                        best_y = trial_y

        return best_x, best_y
