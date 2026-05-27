# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact Differential Evolution (DE) optimizer for black-box minimization.
# Search state: A fixed-size population of candidate solutions (vectors) and their corresponding objective values.
# Candidate generation: For each target vector, a mutant vector is formed using the DE/rand/1 scheme with dither (F scaled by a uniform random number). Crossover is binomial (binary crossover with probability CR).
# Selection and replacement: Greedy selection – if the trial vector has a lower objective value than the target, it replaces the target in the next generation.
# Adaptation: The mutation scale factor F is not static but drawn uniformly from [0.5, 1.0] per individual to provide variety (dither). Crossover probability CR is fixed at 0.9.
# Exploration mechanisms: Dither in F and high CR promote exploration; the population maintains diversity through mutation and crossover.
# Exploitation mechanisms: The greedy selection and the best-so-far tracking focus on reducing objective values. The population contracts over time due to selection pressure.
# Boundary handling: Trial vectors are clipped component-wise to the domain bounds (lower/upper or lb/ub).
# Budget strategy: All evaluations are counted; the budget is split into an initial population evaluation and then iterative generations. The loop stops immediately when the remaining budget reaches zero.
# Closest known influences: Classic DE/rand/1/bin with dither and clipping.
# Novelty or unusual aspects: None – this is a standard, well‑known baseline.
# Failure modes: May converge prematurely on highly multimodal landscapes; may struggle with very high dimensionality if population size is limited by budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # --- Read bounds ---------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot determine bounds from func")
        # ensure correct shape
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)

        # --- DE parameters -------------------------------------------------
        popsize = max(4, min(10 * self.dim, self.budget // 2))
        F_low, F_high = 0.5, 1.0          # dither range
        CR = 0.9
        # --- Initialisation ------------------------------------------------
        pop = lb + (ub - lb) * np.random.uniform(0, 1, (popsize, self.dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_y = np.inf
        evals = 0
        for i in range(popsize):
            if evals >= self.budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()

        # --- Main DE loop --------------------------------------------------
        while evals < self.budget:
            for i in range(popsize):
                if evals >= self.budget:
                    break
                # pick three distinct random indices different from i
                candidates = list(range(popsize))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # mutation with dither: F is drawn per individual
                F = np.random.uniform(F_low, F_high)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # boundary clipping
                trial = np.clip(trial, lb, ub)

                # evaluation
                trial_fitness = func(trial)
                evals += 1

                # selection (minimisation)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
