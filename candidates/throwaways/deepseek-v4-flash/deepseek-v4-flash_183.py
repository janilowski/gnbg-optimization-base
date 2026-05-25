import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a simple black-box minimization algorithm based on
# Differential Evolution (DE/rand/1/bin) with fixed parameters and a cache to avoid
# re-evaluating the same point. It is designed to be robust across dimensions and
# to respect the evaluation budget strictly.
# Search state: A population of candidate solutions stored as a numpy array of shape
# (popsize, dim). The best solution found so far is tracked separately.
# Candidate generation: For each parent, a mutant vector is created by adding the scaled
# difference of two distinct random population members to a third distinct member.
# Then binomial crossover combines the parent and mutant to produce a trial vector.
# Selection and replacement: The trial vector is evaluated (unless cached). If it yields
# a lower objective value than the parent, it replaces the parent. Otherwise the parent
# is retained.
# Adaptation: Fixed scale factor F=0.7 and crossover rate CR=0.9. No dynamic adaptation.
# Exploration mechanisms: The differential mutation and crossover provide exploration
# across the search space. Population diversity is maintained by the random selection
# of distinct individuals for mutation.
# Exploitation mechanisms: Selection pressure favours better solutions; good solutions
# are retained and can serve as bases for further mutations. A simple cache prevents
# redundant evaluations, allowing more unique points to be examined.
# Boundary handling: Trial vectors that fall outside [lb, ub] are reflected back into
# bounds (mirroring) to keep the search within the feasible region.
# Budget strategy: The population size is set as max(4*dim, 20) but capped so that
# at least two full generations can be performed. If the budget is too small for a
# single generation, a random search is conducted. The algorithm stops as soon as the
# evaluation counter reaches the budget.
# Closest known influences: Standard DE/rand/1/bin (Storn and Price, 1997).
# Novelty or unusual aspects: The use of a cache and adaptive population sizing to
# handle very small budgets gracefully.
# Failure modes: For very low budgets relative to dimension, performance degrades to
# random search. Fixed parameters may not suit all functions; no per-dimension scaling.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # Choose population size: at least 20 or 4*dim, but no more than budget/2
        # to allow at least two generations after initialisation.
        self.popsize = max(4 * dim, 20)
        if self.popsize > budget // 2:
            self.popsize = budget // 2
        if self.popsize < 2:
            self.popsize = 2

    def __call__(self, func):
        # ------------------------------------------------------------
        # 1. Read bounds
        # ------------------------------------------------------------
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb, dtype=float)
                ub = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Cannot read bounds from func")

        dim = self.dim
        budget = self.budget
        popsize = self.popsize

        # ------------------------------------------------------------
        # 2. Initialise population (uniform random in [lb, ub])
        # ------------------------------------------------------------
        # Ensure lb, ub are 1D arrays of length dim
        lb = np.broadcast_to(lb, (dim,))
        ub = np.broadcast_to(ub, (dim,))

        pop = lb + np.random.rand(popsize, dim) * (ub - lb)
        # Evaluate initial population
        evals = 0
        cache = {}
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_y = np.inf

        for i in range(popsize):
            if evals >= budget:
                break
            key = tuple(pop[i])
            if key in cache:
                y = cache[key]
            else:
                y = func(pop[i])
                cache[key] = y
                evals += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # ------------------------------------------------------------
        # 3. Main DE loop (rand/1/bin)
        # ------------------------------------------------------------
        F = 0.7          # scale factor
        CR = 0.9         # crossover rate

        while evals < budget:
            # Terminate if we cannot finish at least one full generation
            if evals + popsize > budget:
                break

            for i in range(popsize):
                if evals >= budget:
                    break

                # Choose three distinct random indices != i
                indices = list(range(popsize))
                indices.remove(i)
                np.random.shuffle(indices)
                a, b, c = indices[:3]

                # Mutation
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Crossover (binomial)
                cross_points = np.random.rand(dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                # Boundary handling (reflection)
                # For each coordinate, reflect into bounds
                trial = np.where(trial < lb, 2*lb - trial, trial)
                trial = np.where(trial > ub, 2*ub - trial, trial)
                # Clamp to avoid over-reflection (e.g., if lb >> ub)
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                key = tuple(trial)
                if key in cache:
                    trial_val = cache[key]
                else:
                    trial_val = func(trial)
                    cache[key] = trial_val
                    evals += 1

                # Selection
                if trial_val <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_val
                    if trial_val < best_y:
                        best_y = trial_val
                        best_x = trial.copy()

        # ------------------------------------------------------------
        # 4. Final clean-up: if we still have a few evaluations left,
        #    do random perturbations around the best (local refinement)
        # ------------------------------------------------------------
        while evals < budget:
            # Sample a point near the current best
            step = 0.1 * (ub - lb) * np.random.randn(dim)
            trial = best_x + step
            trial = np.clip(trial, lb, ub)
            key = tuple(trial)
            if key in cache:
                trial_val = cache[key]
            else:
                trial_val = func(trial)
                cache[key] = trial_val
                evals += 1
            if trial_val < best_y:
                best_y = trial_val
                best_x = trial.copy()

        return best_x, best_y
