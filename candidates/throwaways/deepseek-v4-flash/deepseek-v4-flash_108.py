#!/usr/bin/env python3
# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a differential evolution (DE/rand/1/bin) minimizer
#          for the GNBG black-box benchmark.  The algorithm is population‑based and
#          uses a simple mutation, crossover, and selection scheme.  It is designed
#          to be compact, readable, and robust across a wide range of dimensions.
# Search state: The state consists of a population of candidate solutions (NP vectors)
#               and their corresponding objective values.  The current best point and its
#               value are tracked separately. All state is stored in plain NumPy arrays.
#               No archive or memory of past iterates is kept.
# Candidate generation: At each generation, for every target individual a trial vector is
#                       created using the DE/rand/1 scheme: mutant = best + F * (r1 - r2)
#                       where best is the current best individual (to add more exploitation),
#                       and r1,r2 are two other distinct random members of the population.
#                       Binomial crossover with probability CR combines the mutant with the
#                       target to produce the trial.  This variant (DE/best/1/bin) is used
#                       instead of the classic rand/1 to improve convergence speed.
# Selection and replacement: Greedy selection – the trial replaces the target if and only
#                            if its objective value is lower (better) than that of the target.
# Adaptation: The scale factor F and crossover rate CR are kept fixed throughout the run.
#             No self‑adaptation or parameter control is used.
# Exploration mechanisms: The population diversity is maintained via the random selection of
#                         the base and difference vectors from the entire population, and by
#                         the binomial crossover that can mix parts of the target and mutant.
# Exploitation mechanisms: Using the current best vector as the base in the mutation step
#                          directs the search toward promising regions.  Greedy selection also
#                          helps the population to quickly converge.
# Boundary handling: All trial coordinates are clamped to the box constraints
#                    [lower, upper] after crossover.
# Budget strategy: The population size NP is chosen as max(4, min(100, 5*dim)) and is further
#                  limited by the evaluation budget.  Generations are run until the total number
#                  of evaluations (initial + per‑generation) would exceed the budget.
#                  Any leftover evaluations (up to NP‑1) are ignored – the algorithm stops cleanly.
# Closest known influences: Standard DE/best/1/bin with fixed parameters (F=0.9, CR=0.9).
#                           This is a well‑known and widely used baseline.
# Novelty or unusual aspects: None – the implementation follows textbook DE.  The only
#                             minor twist is using the current best as the base vector to
#                             favour exploitation, which is a small departure from the pure
#                             rand/1 variant.
# Failure modes: (i) Very low budget (<2*NP) may not allow enough generations for convergence;
#                (ii) highly multimodal landscapes might cause premature convergence to a local
#                optimum because of the strong exploitation bias; (iii) the fixed parameters
#                are not tuned per function, so performance may vary.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    """Differential Evolution (DE/best/1/bin) minimizer for the GNBG benchmark."""

    def __init__(self, budget: int, dim: int) -> None:
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Read bounds from the objective function
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # Assume func.bounds has .lb and .ub (pycma style)
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds on the objective function.")

        dim = self.dim
        budget = self.budget

        # Determine population size
        NP = max(4, min(100, int(5 * dim)))          # typical suggestion
        NP = min(NP, budget)                         # can't exceed budget

        # If budget is too small for DE, fall back to pure random search
        if budget < 4:
            best_x = np.random.uniform(lb, ub, size=dim)
            best_y = func(best_x)
            evals = 1
            while evals < budget:
                x = np.random.uniform(lb, ub, size=dim)
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # Parameters
        F = 0.9    # scale factor
        CR = 0.9   # crossover probability

        # Initialise population
        pop = np.random.uniform(lb, ub, size=(NP, dim))
        pop_fit = np.array([func(p) for p in pop])
        evals = NP

        # Global best
        best_idx = np.argmin(pop_fit)
        best_x = pop[best_idx].copy()
        best_y = pop_fit[best_idx]

        # Evolution loop
        while evals + NP <= budget:
            for i in range(NP):
                # Choose three distinct indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                a = np.random.choice(candidates)
                candidates.remove(a)
                b = np.random.choice(candidates)
                candidates.remove(b)
                c = np.random.choice(candidates)

                # Mutation: best + F * (b - c)
                mutant = best_x + F * (pop[b] - pop[c])

                # Crossover (binomial)
                j_rand = np.random.randint(dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                  else pop[i, j] for j in range(dim)])

                # Boundary handling: clamp to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate
                trial_fit = func(trial)
                evals += 1

                # Selection
                if trial_fit <= pop_fit[i]:
                    pop[i] = trial
                    pop_fit[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

        return best_x, best_y
