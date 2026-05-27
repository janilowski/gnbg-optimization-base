# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE/rand/1/bin) for black-box minimization.
# Search state: Population of candidate solutions stored in an array of shape (popsize, dim).
# Candidate generation: For each target vector, create a mutant via v = x_r1 + F*(x_r2 - x_r3)
#   where r1, r2, r3 are distinct random indices. Then binomial crossover with probability CR
#   to combine mutant and target into a trial vector. Out‑of‑bound components are reflected
#   back into the domain; if reflection overshoots, coordinates are clamped.
# Selection and replacement: Greedy – if the trial vector yields a lower objective value than
#   the target, the target is replaced in the population.
# Adaptation: Fixed parameters – popsize = min(10*dim, 200), F = 0.8, CR = 0.9. No dynamic
#   adaptation.
# Exploration mechanisms: Mutation uses random differentials, crossover with fairly high CR
#   promotes diversity.
# Exploitation mechanisms: Greedy selection drives the population towards better regions.
# Boundary handling: Reflection (mirror) with clamping.
# Budget strategy: Count evaluations precisely; stop as soon as the budget is exhausted.
# Closest known influences: Classic DE (Storn & Price, 1995).
# Novelty or unusual aspects: None – straightforward, parameter‑fixed DE.
# Failure modes: May converge prematurely on highly multimodal landscapes; fixed
#   parameters may not suit all problem scales or dimensionalities.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ----- read bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot locate bounds from func object")

        # ----- population settings -----
        popsize = min(10 * self.dim, 200)          # reasonable population size
        if popsize < 10:
            popsize = 10
        F = 0.8          # mutation factor
        CR = 0.9         # crossover probability

        # ----- initialisation -----
        pop = lb + np.random.rand(popsize, self.dim) * (ub - lb)  # uniform random
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_y = np.inf
        evals = 0

        # evaluate initial population
        for i in range(popsize):
            if evals >= self.budget:
                break
            y = func(pop[i])
            fitness[i] = y
            evals += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # ----- main loop -----
        while evals < self.budget:
            for i in range(popsize):
                if evals >= self.budget:
                    break

                # choose three distinct random indices different from i
                candidates = list(range(popsize))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # DE/rand/1 mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # binomial crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR,
                                 mutant, pop[i])
                # ensure at least one component from mutant
                trial[j_rand] = mutant[j_rand]

                # boundary handling: reflect, then clamp if still out
                # reflection: out_of_low = trial < lb -> lb + (lb - trial)
                # out_of_high = trial > ub -> ub - (trial - ub)
                low_idx = trial < lb
                high_idx = trial > ub
                trial[low_idx] = lb[low_idx] + (lb[low_idx] - trial[low_idx])
                trial[high_idx] = ub[high_idx] - (trial[high_idx] - ub[high_idx])
                # clamp any remaining outliers (numerical safety)
                trial = np.clip(trial, lb, ub)

                # evaluate trial
                y_trial = func(trial)
                evals += 1

                # greedy selection
                if y_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = y_trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()

        return best_x, best_y
