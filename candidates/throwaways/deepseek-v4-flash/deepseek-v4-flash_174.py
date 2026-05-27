# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact Differential Evolution (DE/rand/1/bin) 
# algorithm for black-box minimization. It is designed to work robustly across 
# dimensions and budgets, using only numpy and the Python standard library. 
# The algorithm respects a given evaluation budget and handles bound constraints 
# by clamping coordinates.
# Search state: A population of candidate solutions (dimension vectors) and their 
# associated objective values. The population is stored as a 2D numpy array and 
# fitness as a 1D array. The best known solution is tracked separately.
# Candidate generation: For each target vector, a mutant vector is created by 
# adding scaled difference of two random population members to a third (randomly 
# chosen) member. Then, binomial crossover combines the mutant with the target to 
# produce a trial vector.
# Selection and replacement: Greedy one-to-one selection: if the trial vector 
# yields a lower (or equal) objective value than the target, it replaces the target.
# Adaptation: Fixed parameters (F = 0.8, CR = 0.9). No dynamic adaptation.
# Exploration mechanisms: The differential mutation step size (controlled by F) 
# and random selection of base and difference vectors maintain diversity.
# Exploitation mechanisms: Selection pressure (replacement of inferior solutions) 
# and the recombination of trial vectors with current population guide the search 
# toward promising regions.
# Boundary handling: Trial vectors are clamped to the specified lower and upper 
# bounds after crossover.
# Budget strategy: The population size is chosen based on the budget (max(5, min(50, 
# budget//5))) to allow several generations. Each generation evaluates exactly 
# pop_size new trial vectors. If the budget is too small for a full generation, 
# a random search is performed instead.
# Closest known influences: Standard Differential Evolution (DE/rand/1/bin) as 
# described by Storn and Price (1997).
# Novelty or unusual aspects: None; the implementation follows the classic DE 
# design with a simple budget‑aware population size rule.
# Failure modes: May converge prematurely on highly multimodal, rugged landscapes 
# if the population loses diversity. Fixed parameters may be suboptimal for 
# specific problems. Performance degrades if the budget is too low to allow 
# meaningful search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Black-box minimization using Differential Evolution (DE/rand/1/bin).
    """
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # 1. Determine bounds from func
        try:
            lb = getattr(func, 'lower')
            ub = getattr(func, 'upper')
        except AttributeError:
            try:
                lb = func.bounds.lb
                ub = func.bounds.ub
            except AttributeError:
                raise AttributeError("Function lacks lower/upper or bounds.lb/ub attributes.")

        # 2. Budget management
        budget = self.budget
        dim = self.dim

        # Population size: at least 5, at most 50, and ensures at least 2 generations possible
        if budget < 5:
            pop_size = 1
        else:
            pop_size = max(5, min(50, budget // 5))
        # Ensure at least one generation after initial evaluation
        if pop_size > budget // 2:
            pop_size = max(1, budget // 2)

        # 3. Initialization
        # Random population within bounds
        pop = np.random.uniform(low=lb, high=ub, size=(pop_size, dim))
        # Evaluate initial population
        fitness = np.array([func(x) for x in pop])
        evals = pop_size
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # If budget is tiny, just return best from initial sample
        if budget <= pop_size:
            return best_x, best_y

        # 4. DE parameters
        F = 0.8
        CR = 0.9

        # 5. Main loop
        while evals + pop_size <= budget:
            # For each target vector in population
            for i in range(pop_size):
                # Choose three distinct random indices different from i
                idxs = [j for j in range(pop_size) if j != i]
                r = np.random.choice(idxs, size=3, replace=False)
                a, b, c = r[0], r[1], r[2]

                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Boundary handling: clamp to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # Selection
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()
            # If not enough budget left for a full generation, stop
            if evals + pop_size > budget:
                break

        return best_x, best_y
