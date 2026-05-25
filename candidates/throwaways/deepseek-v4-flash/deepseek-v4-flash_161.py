# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A differential evolution (DE) minimizer for black-box functions.
# Search state: A population of vectors (np.ndarray of shape (popsize, dim)).
# Candidate generation: For each target vector, three distinct random population members
#   are selected, and a mutant vector is formed using DE/rand/1:
#   mutant = base + F * (diff1 - diff2).
# Crossover: Binomial crossover between mutant and target, controlled by CR,
#   producing a trial vector.
# Selection and replacement: Greedy: trial replaces target if its objective value is
#   (strictly) lower (minimization). Best-so-far solution is tracked.
# Adaptation: None; F and CR are static (0.8 and 0.9).
# Exploration mechanisms: Mutation via random difference vectors and crossover
#   provide global exploration. Population diversity is maintained through
#   stochastic selection of base and difference vectors.
# Exploitation mechanisms: Greedy selection and the recombination mechanism
#   (if trial is better it replaces the parent) drive local refinement.
# Boundary handling: Trial vectors are clipped to the domain [lower, upper].
# Budget strategy: Population size is set adaptively so that no more than the
#   budgeted evaluations are consumed. The algorithm stops when evaluations reach
#   the budget. Generations are run until the budget is exhausted, possibly with
#   an incomplete final generation.
# Closest known influences: Classic DE/rand/1/bin (Storn & Price 1997).
# Novelty or unusual aspects: None; straightforward implementation with fixed
#   parameters and a simple population size heuristic based on dimension and budget.
# Failure modes: May converge prematurely on highly multimodal landscapes or
#   if the budget is too small relative to dimension. Clip-based boundary
#   handling can cause stagnation near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # Determine problem bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.atleast_1d(np.asarray(func.lower, dtype=float))
            upper = np.atleast_1d(np.asarray(func.upper, dtype=float))
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lower = np.atleast_1d(np.asarray(b.lb, dtype=float))
            upper = np.atleast_1d(np.asarray(b.ub, dtype=float))
        else:
            raise AttributeError("Function must provide bounds via .lower/.upper or .bounds.lb/.bounds.ub")
        # Ensure arrays have correct dimension
        if lower.ndim == 0:
            lower = np.full(self.dim, lower)
            upper = np.full(self.dim, upper)
        if lower.shape[0] != self.dim:
            lower = np.full(self.dim, lower[0])
            upper = np.full(self.dim, upper[0])

        # Population size heuristic: at least 4, not more than 50, and such that
        # at least two full generations can be evaluated (but respecting budget).
        # For DE, typical popsize ~ 10*dim, but we limit to avoid too large overhead.
        popsize = min(50, max(4, self.budget // (self.dim + 1)))
        # Ensure we can run at least one generation (popsize evaluations)
        if popsize > self.budget:
            popsize = max(2, self.budget)  # at least 2 for mutation, but budget small
        # Initialize population uniformly in bounds
        pop = lower + np.random.rand(popsize, self.dim) * (upper - lower)
        # Evaluate initial population
        evals = 0
        fitness = np.empty(popsize)
        for i in range(popsize):
            fitness[i] = func(pop[i])
            evals += 1
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # DE parameters
        F = 0.8      # mutation factor
        CR = 0.9     # crossover probability

        # Main loop: run generations until budget exhausted
        while evals < self.budget:
            # Number of trials we can afford this generation (at most popsize)
            trials_remaining = self.budget - evals
            trials_this_gen = min(popsize, trials_remaining)
            if trials_this_gen == 0:
                break
            # For each target in the generation (first trials_this_gen individuals)
            for i in range(trials_this_gen):
                # Select three distinct random indices different from i
                indices = list(range(popsize))
                indices.remove(i)
                a, b, c = np.random.choice(indices, size=3, replace=False)
                # Mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # Boundary clipping
                trial = np.clip(trial, lower, upper)
                # Evaluate trial
                trial_fit = func(trial)
                evals += 1
                # Selection
                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()
                # Check budget after each evaluation
                if evals >= self.budget:
                    break
            # If we only processed part of the population, the rest remain unchanged
        return best_x, best_y
