# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A classic Differential Evolution (DE) algorithm minimising a black-box function
#   within a fixed evaluation budget. The implementation uses the DE/rand/1/bin variant.
# Search state: A population of candidate solutions stored in a 2D numpy array (size NP x dim),
#   along with their fitness values (1D array). The best solution and its fitness are tracked.
# Candidate generation: For each target vector in the population, three distinct random vectors
#   are selected (different from the target and each other). A mutant is created as
#   base + F * (diff1 - diff2). Crossover with the target vector using binomial (uniform)
#   crossover with probability CR produces a trial vector. At least one dimension is taken from
#   the mutant.
# Selection and replacement: The trial vector replaces the target vector in the population if and
#   only if it yields a lower (better) objective value (minimization).
# Adaptation: Fixed parameters F (0.8) and CR (0.9). No self-adaptation or parameter control.
# Exploration mechanisms: The differential mutation perturbation promotes exploration, especially
#   early in the run. Crossover blends target and mutant components.
# Exploitation mechanisms: As the population converges, the differences between vectors become
#   smaller, leading to finer-grained local search (implicit exploitation). The greedy selection
#   (replacement only on improvement) drives exploitation.
# Boundary handling: Trial vectors that violate bounds are clipped (clamped) to the nearest bound.
#   This ensures all evaluated points are feasible.
# Budget strategy: The algorithm evaluates all NP initial points first, then runs generational
#   cycles evaluating exactly NP trial vectors per generation until the budget is exhausted.
#   Evaluations are counted individually; the loop terminates as soon as the budget is reached,
#   even in the middle of a generation.
# Closest known influences: Standard Differential Evolution (DE/rand/1/bin) as described by
#   Storn and Price (1997).
# Novelty or unusual aspects: None. This is a straightforward baseline implementation designed
#   for clarity and robustness across dimensions.
# Failure modes: Premature convergence when the population collapses before the optimum is found;
#   stagnation when F is too small; high F can cause excessive exploration near boundaries.
#   Clipping to bounds can artificially create duplicate solutions on the boundary.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # DE parameters – fixed for simplicity
        self.F = 0.8
        self.CR = 0.9
        # Population size – moderate relation to dimension
        self.NP = max(4 * dim, 10)

    def __call__(self, func):
        # Read bounds from the function object
        try:
            lb = np.atleast_1d(np.asarray(func.lower, dtype=float))
            ub = np.atleast_1d(np.asarray(func.upper, dtype=float))
        except AttributeError:
            # Alternative: func.bounds.lb / func.bounds.ub
            lb = np.atleast_1d(np.asarray(func.bounds.lb, dtype=float))
            ub = np.atleast_1d(np.asarray(func.bounds.ub, dtype=float))
        # Broadcast to full dimension if scalars given
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)
        lb = lb.astype(float)
        ub = ub.astype(float)

        dim = self.dim
        NP = self.NP
        F = self.F
        CR = self.CR
        budget = self.budget

        # Initialise population uniformly in the search space
        pop = np.random.uniform(lb, ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_x = None
        best_y = np.inf
        evals = 0

        # Evaluate initial population
        for i in range(NP):
            if evals >= budget:
                break
            y = func(pop[i])
            fitness[i] = y
            evals += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Main DE loop
        while evals < budget:
            # For each target vector, generate a trial
            for i in range(NP):
                if evals >= budget:
                    break

                # Select three distinct indices different from i
                indices = list(range(NP))
                indices.remove(i)
                a, b, c = np.random.choice(indices, size=3, replace=False)

                # Mutation: base = pop[a], difference = pop[b] - pop[c]
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Crossover (binomial)
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary handling – clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                y_trial = func(trial)
                evals += 1

                # Selection – replace if better (minimization)
                if y_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = y_trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()

        return best_x, best_y
