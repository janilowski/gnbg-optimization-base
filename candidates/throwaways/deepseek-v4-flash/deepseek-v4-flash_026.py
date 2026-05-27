# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implementation of a standard Differential Evolution (DE/rand/1/bin) for black-box minimization.
# Search state: A population of candidate solutions (vectors) and their associated fitness values.
# Candidate generation: For each target vector, three distinct random population members are selected. A mutant vector is created as base + F * (diff1 - diff2). Then binomial crossover with probability CR combines the target and mutant to produce a trial vector.
# Selection and replacement: The trial vector replaces the target if its fitness is not worse (minimization).
# Adaptation: The algorithm uses fixed parameters: scaling factor F and crossover probability CR. No adaptive mechanisms.
# Exploration mechanisms: Global exploration via random selection of base and difference vectors; crossover allows mixing of dimensions.
# Exploitation mechanisms: Population converges over generations as better solutions replace worse ones; greedy selection.
# Boundary handling: Trial vectors are clipped to the bounds if components exceed them.
# Budget strategy: The algorithm uses the entire budget of function evaluations. The population size is set as min(50, budget // 5) to allow multiple generations. The algorithm stops when evaluations exceed budget, even mid-generation.
# Closest known influences: Classic Differential Evolution (Storn and Price, 1997).
# Novelty or unusual aspects: None; straightforward implementation for benchmarking.
# Failure modes: May struggle on highly multimodal or deceptive landscapes with noisy evaluations (GNBG). Fixed parameters may not suit all problems. Clipping bounds may cause stagnation near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot find bounds from the provided function.")
        # Ensure they are 1D arrays
        lb = np.ravel(lb)
        ub = np.ravel(ub)

        # DE parameters
        F = 0.8          # scaling factor
        CR = 0.9         # crossover probability

        # Population size: at least 4 (requires 3 distinct indices for mutation), at most budget/5 or 50
        pop_size = max(4, min(50, self.budget // 5))
        # If budget is too small for even one generation, do random search
        if self.budget < pop_size:
            best_x = np.random.uniform(lb, ub)
            best_y = func(best_x)
            for _ in range(self.budget - 1):
                cand = np.random.uniform(lb, ub)
                val = func(cand)
                if val < best_y:
                    best_y = val
                    best_x = cand
            return best_x, best_y

        # Initialize population
        population = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            fitness[i] = func(population[i])
        evals_used = pop_size
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]

        # Main loop
        while evals_used < self.budget:
            for i in range(pop_size):
                if evals_used >= self.budget:
                    break

                # Select three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    # Not enough individuals; skip or break
                    break
                idx = np.random.choice(candidates, 3, replace=False)
                a, b, c = population[idx[0]], population[idx[1]], population[idx[2]]

                # Mutation: DE/rand/1
                mutant = a + F * (b - c)

                # Crossover: binomial
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])
                # Ensure at least one component comes from mutant (jitter not needed)
                # Boundary clipping
                trial = np.clip(trial, lb, ub)

                # Evaluation
                trial_fitness = func(trial)
                evals_used += 1

                # Selection: greedy
                if trial_fitness <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

            # Update best from population after full generation (if any new best found)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_y:
                best_y = fitness[best_idx]
                best_x = population[best_idx].copy()

        return best_x, best_y
