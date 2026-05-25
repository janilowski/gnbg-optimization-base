# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a Differential Evolution (DE) optimizer using the rand/1/bin strategy with fixed parameters (F=0.8, CR=0.9). It is designed for black-box minimization under a strict evaluation budget. The population size is scaled with the budget, and leftover evaluations are used for a simple local search around the best solution found.
# Search state: A population of candidate solutions (numpy array of shape [pop_size, dim]) and their corresponding fitness values. The algorithm also tracks the best solution and its fitness, plus an evaluation counter.
# Candidate generation: For each parent, a trial vector is formed by DE/rand/1 mutation (donor = a + F*(b-c)) followed by binomial crossover with the parent. Three distinct random individuals are chosen from the population (different from the parent).
# Selection and replacement: Greedy selection: the trial replaces the parent if its fitness is <= the parent's fitness. The global best is updated if an improved fitness is found.
# Adaptation: None – F and CR remain constant throughout the run. Population size is determined once based on budget.
# Exploration mechanisms: Random initial population; differential mutation (difference of two other individuals) provides random directed steps; binomial crossover allows mixing of coordinates; leftover budget local search uses Gaussian perturbations around the best solution.
# Exploitation mechanisms: Greedy replacement ensures that better solutions are retained; the local search at the end refines the best point.
# Boundary handling: Simple clipping of all generated points to the lower and upper bounds.
# Budget strategy: Population size set as max(4, min(50, budget//10)) but capped so that at least one generation is possible. The number of full generations is (budget - pop_size) // pop_size. After the generational loop, any remaining evaluations are spent on local search steps from the best solution.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997) with rand/1/bin, clipping boundary handling, and a final local search (similar to a simple random walk hill climber).
# Novelty or unusual aspects: Very simple implementation with minimal parameter adaptation; budget-driven population size and a post-run local search to use all available evaluations.
# Failure modes: Fixed F and CR may not be optimal for all landscapes; clipping can cause loss of diversity (population may collapse); on highly multimodal functions, the algorithm may converge prematurely; very low budgets may limit the effectiveness of the population-based stage.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # -------- get bounds --------
        if hasattr(func, 'lower'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        else:
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        # safety check (single value vs. array)
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)

        budget = self.budget
        dim = self.dim

        # -------- handle very small budgets --------
        if budget < 4:
            # simple random search
            best_x = np.random.uniform(lb, ub, size=dim)
            best_y = func(best_x)
            evals = 1
            while evals < budget:
                x = np.random.uniform(lb, ub, size=dim)
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x
            return best_x, best_y

        # -------- determine population size --------
        # scale with budget but keep between 4 and 50
        N = max(4, min(50, budget // 10))
        # ensure at least one generation possible: initial+N ≤ budget
        if budget < 2 * N:
            N = max(4, budget // 2)   # shrink so that we can have initial + 1 generation
        # initialise population
        pop = np.random.uniform(lb, ub, size=(N, dim))
        fitness = np.array([func(p) for p in pop])
        evals = N

        # best so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # -------- DE parameters (fixed) --------
        F = 0.8
        CR = 0.9

        # number of full generations we can run
        max_gen = (budget - N) // N   # each generation costs exactly N evaluations

        # -------- generational loop --------
        for _ in range(max_gen):
            # iterate over population members
            for i in range(N):
                # pick three distinct random indices ≠ i
                indices = list(range(N))
                indices.remove(i)
                a, b, c = np.random.choice(indices, size=3, replace=False)

                # mutation
                donor = pop[a] + F * (pop[b] - pop[c])
                # binomial crossover
                cross_mask = np.random.rand(dim) < CR
                if not cross_mask.any():          # ensure at least one coordinate changes
                    cross_mask[np.random.randint(dim)] = True
                trial = np.where(cross_mask, donor, pop[i])

                # boundary clipping
                trial = np.clip(trial, lb, ub)

                # evaluation
                trial_fit = func(trial)
                evals += 1

                # selection
                if trial_fit <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

            # (the loop stops exactly at budget thanks to max_gen)

        # -------- leftover evaluations: local search around best --------
        remaining = budget - evals
        step_scale = 0.02 * (ub - lb)   # size relative to domain width
        for _ in range(remaining):
            # small Gaussian step
            perturbation = np.random.randn(dim) * step_scale
            candidate = best_x + perturbation
            candidate = np.clip(candidate, lb, ub)
            y = func(candidate)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = candidate

        return best_x, best_y
