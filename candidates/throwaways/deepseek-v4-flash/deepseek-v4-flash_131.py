# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact implementation of Differential Evolution (DE) with the
#   classic DE/rand/1/bin scheme. Mutation scaling factor F is drawn uniformly
#   from [0.5, 1.0] each generation to blend exploration and exploitation.
# Search state: A population of NP candidate vectors and their corresponding
#   fitness values. The best individual seen so far is tracked separately.
# Candidate generation: For each target vector, three distinct random vectors
#   are selected from the population. A mutant is created as base + F*(diff1 - diff2)
#   (DE/rand/1). A trial vector is formed via binomial crossover with probability CR.
# Selection and replacement: Greedy: if trial vector yields lower fitness, it
#   replaces the target in the next generation.
# Adaptation: The scaling factor F is randomized per generation; CR remains fixed.
# Exploration mechanisms: Random mutation, crossover, and bound reflection.
# Exploitation mechanisms: Selection pressure that retains better individuals;
#   smaller F values (toward 0.5) induce more local search.
# Boundary handling: Reflective correction – any coordinate that violates bounds
#   is reflected back into the feasible interval.
# Budget strategy: Each call to func counts one evaluation; the algorithm stops
#   when the total number of evaluations reaches the budget.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: Minimalistic, fully deterministic except for
#   random components, and uses no external libraries beyond numpy and standard
#   library.
# Failure modes: May converge prematurely on highly multimodal landscapes if
#   population size is too small. With a fixed max budget, performance degrades
#   when dim is very high (>50) relative to budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Extract bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.array(b.lb, dtype=float)
            ub = np.array(b.ub, dtype=float)
        else:
            raise ValueError("Function must provide bounds via .lower/.upper or .bounds.lb/.bounds.ub")

        dim = self.dim
        # Ensure lb/ub are 1D arrays of length dim
        if lb.ndim == 0:
            lb = np.full(dim, lb)
            ub = np.full(dim, ub)

        # DE parameters
        NP = max(4, min(50, 10 * dim))           # population size
        CR = 0.9                                 # crossover rate (fixed)
        # F will be drawn per generation: F ~ U[0.5, 1.0]

        # Initialise population uniformly in bounds
        pop = lb + np.random.rand(NP, dim) * (ub - lb)
        # Evaluate initial population
        fitness = np.array([func(p) for p in pop])   # NP evaluations
        evals = NP

        # Keep track of best so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        while evals < self.budget:
            # Number of evaluations we can still perform (at most NP per generation)
            remaining = self.budget - evals
            if remaining <= 0:
                break

            # We will process at most 'remaining' target vectors this generation
            num_targets = min(NP, remaining)

            # Randomise scaling factor for this generation
            F = 0.5 + np.random.rand() * 0.5   # U[0.5, 1.0]

            # Shuffle indices to avoid bias in target order
            indices = np.random.permutation(NP)

            for i in range(num_targets):
                target_idx = indices[i]
                target = pop[target_idx].copy()
                target_fit = fitness[target_idx]

                # Choose three distinct random indices different from target_idx
                r = np.random.choice([j for j in range(NP) if j != target_idx], size=3, replace=False)
                a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]

                # Mutation: DE/rand/1
                mutant = a + F * (b - c)

                # Binomial crossover
                trial = target.copy()
                # Select at least one coordinate to cross
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary reflection (reflective)
                trial = np.where(trial < lb, 2 * lb - trial, trial)
                trial = np.where(trial > ub, 2 * ub - trial, trial)
                # Clamp to ensure no overshoot (due to floating point)
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_fit = func(trial)
                evals += 1

                # Greedy selection
                if trial_fit < target_fit:
                    pop[target_idx] = trial
                    fitness[target_idx] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

        return best_x, best_y
