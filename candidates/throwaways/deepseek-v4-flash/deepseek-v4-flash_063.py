# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Differential Evolution (DE) algorithm for black-box minimization.
# Search state: A population of candidate solutions uniformly distributed within bounds.
# Candidate generation: For each target vector, a trial vector is created via DE/rand/1/bin:
#   mutant = base + F * (diff1 - diff2), then binomial crossover with target.
# Selection and replacement: Deterministic replacement if trial has better (lower) objective value.
# Adaptation: None fixed parameters F=0.9, CR=0.9.
# Exploration mechanisms: Differential mutation allows global exploration; crossover blends solutions.
# Exploitation mechanisms: Selection pressure towards better solutions; as population converges, step sizes shrink naturally.
# Boundary handling: Trial vectors are clipped to the search domain.
# Budget strategy: Population size is chosen as min(10 * dim, budget // 2, 100) to leave room for generations.
# Closest known influences: Classic DE/rand/1/bin (Storn & Price, 1997).
# Novelty or unusual aspects: No adaptation or archive; simple clipping.
# Failure modes: May stagnate in rugged or separable landscapes; small population for low budget.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np
import math

class Algorithm:
    """
    Differential Evolution for GNBG benchmark.
    """
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        # Population size: at least 4, at most min(10*dim, budget/2, 100).
        popsize = min(10 * self.dim, self.budget // 2, 100)
        if popsize < 4:
            popsize = max(4, self.budget)  # fallback for tiny budget
        self.popsize = popsize

        self.F = 0.9       # mutation factor
        self.CR = 0.9      # crossover probability

    def __call__(self, func):
        # read bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and func.bounds is not None:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # fallback to unit hypercube
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)

        # ensure dimensions match
        lb = lb.ravel()
        ub = ub.ravel()
        if len(lb) != self.dim:
            lb = np.full(self.dim, lb[0] if lb.size else 0.0)
        if len(ub) != self.dim:
            ub = np.full(self.dim, ub[0] if ub.size else 1.0)

        # Initialise population uniformly within bounds
        pop = np.random.uniform(lb, ub, size=(self.popsize, self.dim))

        evals = 0
        best_x = None
        best_y = float('inf')

        # Evaluate initial population
        for i in range(self.popsize):
            y = func(pop[i])
            evals += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()
            if evals >= self.budget:
                # budget exhausted during initialization
                return best_x, best_y

        # DE main loop
        while evals < self.budget:
            new_pop = np.empty_like(pop)
            for i in range(self.popsize):
                # select three distinct random indices, none equal to i
                candidates = list(range(self.popsize))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                # mutation: base = a, difference = b - c
                mutant = pop[a] + self.F * (pop[b] - pop[c])

                # binomial crossover
                trial = pop[i].copy()
                # at least one component from mutant
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                # clip to bounds
                trial = np.clip(trial, lb, ub)

                # evaluate trial
                y_trial = func(trial)
                evals += 1

                # selection
                if y_trial < func(pop[i]):  # compare with current objective (minimization)
                    new_pop[i] = trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()
                else:
                    new_pop[i] = pop[i].copy()

                if evals >= self.budget:
                    # budget exhausted during generation
                    # finish remaining assignments to new_pop (unused) but we break
                    # copy remaining from old pop for consistency
                    for k in range(i+1, self.popsize):
                        new_pop[k] = pop[k].copy()
                    pop = new_pop
                    return best_x, best_y

            pop = new_pop

        return best_x, best_y
