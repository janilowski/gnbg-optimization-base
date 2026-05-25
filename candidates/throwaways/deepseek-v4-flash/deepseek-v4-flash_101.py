# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Standard Differential Evolution (DE/rand/1/bin) for black-box minimization.
#          Population-based, uses mutation and binomial crossover to generate trial vectors.
#          Simple, robust, and dimension-independent.
# Search state: A population of candidate solutions (real vectors) stored as a 2D numpy array
#               (popsize x dim) along with their fitness values.
# Candidate generation: For each population member, a mutant vector is created by adding
#                       the scaled difference of two randomly selected distinct vectors to
#                       a third random vector (DE/rand/1). Then binomial crossover with the
#                       original candidate produces a trial vector.
# Selection and replacement: Greedy selection: if the trial vector has lower (better) fitness,
#                            it replaces the original in the next generation.
# Adaptation: None; mutation scaling factor F and crossover rate CR are fixed.
# Exploration mechanisms: Large population size (10×dim) and high crossover rate (0.7) keep
#                         diversity early. Mutation differences promote global search.
# Exploitation mechanisms: Greedy selection and decreasing population diversity over time
#                          (if convergence occurs) help fine-tune solutions.
# Boundary handling: Trial vectors are clipped to the search bounds (lower, upper) after generation.
# Budget strategy: Evaluation budget is divided into initial population evaluation plus a fixed
#                  number of full generations (generations = (budget - popsize) // popsize).
#                  If budget is very small, only random sampling is performed.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None (standard implementation).
# Failure modes: May converge prematurely on highly multimodal problems if F or CR are
#                unsuitable; fixed parameters are not adapted per problem.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ---- extract bounds ----
        try:
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lower = np.asarray(func.bounds.lb, dtype=float)
                upper = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Cannot determine bounds from func.")

        if lower.shape != (self.dim,) or upper.shape != (self.dim,):
            lower = np.full(self.dim, lower) if np.isscalar(lower) else lower
            upper = np.full(self.dim, upper) if np.isscalar(upper) else upper

        budget = self.budget
        dim = self.dim
        # ---- algorithm parameters ----
        # population size: try to have at least 3*dim, but not exceed half the budget
        popsize = min(max(3 * dim, 10), budget // 2)
        if popsize < 2:
            popsize = 2
        # number of full generations (each generation evaluates popsize new points)
        if popsize < budget:
            gens = (budget - popsize) // popsize
        else:
            gens = 0

        F = 0.8          # mutation scaling factor
        CR = 0.7         # crossover rate

        # ---- initialisation ----
        pop = np.random.uniform(lower, upper, size=(popsize, dim))
        fitness = np.array([func(x) for x in pop])   # initial evaluations
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]
        evals = popsize

        # ---- main loop ----
        for _ in range(gens):
            for i in range(popsize):
                if evals >= budget:
                    break
                # select three distinct random indices different from i
                idxs = [j for j in range(popsize) if j != i]
                np.random.shuffle(idxs)
                a, b, c = idxs[:3]
                # mutant
                mutant = pop[a] + F * (pop[b] - pop[c])
                # binomial crossover
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                # boundary clipping
                trial = np.clip(trial, lower, upper)
                # evaluate
                trial_fit = func(trial)
                evals += 1
                # greedy selection
                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()
            if evals >= budget:
                break

        # if budget is extremely small (only random initialisation)
        # best_x, best_y already hold the best from initial population.

        return best_x, best_y
