# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a classic Differential Evolution (DE/rand/1/bin) optimizer for black-box minimization,
#          with population size adapted to budget and dimension, and uniform random initialization within bounds.
# Search state: Population of candidate solutions (array of shape (popsize, dim)), corresponding fitnesses,
#               and best found so far (best_x, best_y).
# Candidate generation: For each member of the population (target vector), a mutant vector is created by adding
#                       the weighted difference of two randomly selected distinct population members to a third
#                       distinct member. Then binomial crossover produces a trial vector by mixing components from
#                       the mutant and target with probability CR.
# Selection and replacement: Greedy selection: if trial vector yields a lower (better) objective value than the
#                            target, it replaces the target in the population immediately (sequential replacement).
#                            The best overall solution is updated whenever a better trial is found.
# Adaptation: Uses fixed scale factor F=0.8 and crossover rate CR=0.9. No parameter adaptation to maintain
#             simplicity and robustness.
# Exploration mechanisms: Mutation using random differential vectors provides exploration; high CR mixes many
#                         components, encouraging diversity; uniform random initialization covers the search space.
# Exploitation mechanisms: Greedy selection favors better solutions, causing convergence; the differential
#                          recombination refines existing good solutions; the fixed population size maintains pressure.
# Boundary handling: After mutation and before evaluation, each coordinate of the trial vector is clipped to
#                    the [lower, upper] bounds.
# Budget strategy: The algorithm tracks the number of function evaluations used. It initializes and then runs
#                  generations until the budget is exhausted. If the remaining budget is less than the population
#                  size, the algorithm switches to pure random search (only for very small budgets).
# Closest known influences: Standard Differential Evolution (Storn and Price, 1997) with DE/rand/1/bin strategy.
# Novelty or unusual aspects: None; implementation follows standard textbook DE with clipping and simple budget control.
# Failure modes: May converge prematurely if population loses diversity rapidly; high CR can cause excessive
#                disruption in smooth landscapes; fixed F may be suboptimal for certain landscapes; clipped
#                boundaries bias search toward corners; not suitable for highly multimodal problems without
#                additional niching; poor performance on very small budgets (less than about 10*dim evaluations).
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Differential Evolution optimizer for black-box minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        budget = self.budget
        dim = self.dim

        # ---- read bounds ----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        else:
            # assume func has .bounds with .lb and .ub
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        # ensure shape (dim,)
        lb = lb.ravel()
        ub = ub.ravel()

        # ---- handle tiny budget: pure random search ----
        if budget < 5:
            best_y = np.inf
            best_x = None
            for _ in range(budget):
                x = lb + np.random.random(dim) * (ub - lb)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # ---- DE parameters ----
        F = 0.8          # scale factor
        CR = 0.9         # crossover probability
        # population size: at least 4, at most 50, scaled with dim and budget
        popsize = max(4, min(50, int(budget / 10), 10 * dim))
        # ensure we have enough budget for at least one generation after initialization
        if popsize * 2 > budget:
            # fallback to random search if not enough for one full generation
            # (popsize initial evaluations + popsize trials = 2*popsize)
            best_y = np.inf
            best_x = None
            for _ in range(budget):
                x = lb + np.random.random(dim) * (ub - lb)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # ---- initialize population ----
        pop = lb + np.random.uniform(size=(popsize, dim)) * (ub - lb)
        fit = np.full(popsize, np.inf)
        best_y = np.inf
        best_x = None
        evals = 0

        for i in range(popsize):
            if evals >= budget:
                break
            y = func(pop[i])
            fit[i] = y
            evals += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # ---- main DE loop ----
        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break
            # we may not have enough budget for a full generation; process one trial per target
            for i in range(popsize):
                if evals >= budget:
                    break
                # select three distinct random indices different from i
                candidates = list(range(popsize))
                candidates.remove(i)
                r = np.random.choice(candidates, size=3, replace=False)
                a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]
                # mutation
                mutant = a + F * (b - c)
                # binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # boundary clipping
                trial = np.clip(trial, lb, ub)
                # evaluate
                y_trial = func(trial)
                evals += 1
                if y_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = y_trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()
                # if budget exhausted after this evaluation, break out of inner loop
                if evals >= budget:
                    break

        return best_x, best_y
