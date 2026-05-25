import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a Differential Evolution (DE) optimizer
#          designed for the GNBG black-box minimization benchmark. It uses
#          the DE/rand/1/bin variant with fixed parameters suitable for a wide
#          range of continuous optimization problems.
# Search state: A population of candidate solutions (real-valued vectors) stored
#               as a 2D NumPy array of shape (pop_size, dim). The best solution
#               and its objective value are tracked independently.
# Candidate generation: For each population member, a mutant vector is created
#                       by adding the scaled difference of two random distinct
#                       population members to a third distinct random member
#                       (DE/rand/1). Then binomial crossover mixes the mutant
#                       with the current member to form a trial vector.
# Selection and replacement: Greedy selection: a trial vector replaces the
#                            current member if and only if its objective value
#                            is better (lower). The overall best solution is
#                            updated whenever a new best is found.
# Adaptation: No online adaptation of parameters (F, CR, pop_size). The
#             population size is set as a function of dimension and budget.
# Exploration mechanisms: Mutation using scaled random differences and
#                         crossover encourage exploration of the search space.
#                         Random initial population covers the domain.
# Exploitation mechanisms: Greedy selection retains improving solutions.
#                          As generations progress, the population converges
#                          and the scaled differences shrink, leading to local
#                          refinement. No separate local search is used.
# Boundary handling: Trial vectors that violate bounds are clipped to the
#                    nearest bound value.
# Budget strategy: The population size is chosen so that a reasonable number
#                  of generations can be run without exceeding the budget.
#                  After initialization, each generation uses one evaluation
#                  per population member. The loop stops when the budget is
#                  exhausted. The total evaluations never exceed budget.
# Closest known influences: Classic Differential Evolution (Storn & Price,
#                          1997) with DE/rand/1/bin strategy.
# Novelty or unusual aspects: None. A straightforward implementation tailored
#                             to the benchmarking interface.
# Failure modes: May perform poorly on highly multimodal or deceptive
#                landscapes if the population converges prematurely. High-
#                dimensional problems may require larger populations than
#                the budget allows, limiting exploration.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.atleast_1d(np.asarray(func.lower, dtype=float))
            upper = np.atleast_1d(np.asarray(func.upper, dtype=float))
        elif hasattr(func, 'bounds'):
            lower = np.atleast_1d(np.asarray(func.bounds.lb, dtype=float))
            upper = np.atleast_1d(np.asarray(func.bounds.ub, dtype=float))
        else:
            raise ValueError("Objective function has no accessible bounds.")

        dim = self.dim
        # Ensure bounds are broadcastable
        if lower.ndim == 0:
            lower = np.full(dim, lower)
        if upper.ndim == 0:
            upper = np.full(dim, upper)
        lower = lower.reshape(dim)
        upper = upper.reshape(dim)

        # Parameters
        # Population size: at least 4, scaled with dimension, limited by budget
        pop_size = max(4, min(10 * dim, self.budget // 2))
        if pop_size < 4:
            pop_size = 4  # DE requires at least 4

        F = 0.8  # scaling factor
        CR = 0.9  # crossover probability

        budget = self.budget

        # Initialize population uniformly in bounds
        pop = np.random.uniform(lower, upper, size=(pop_size, dim))
        # Evaluate initial population
        fitness = np.empty(pop_size)
        for i in range(pop_size):
            if budget <= 0:
                break
            fitness[i] = func(pop[i])
            budget -= 1
        else:
            # only reached if all evaluations used within loop
            pass

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        while budget > 0:
            for i in range(pop_size):
                if budget <= 0:
                    break

                # Choose three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r = np.random.choice(candidates, size=3, replace=False)
                a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]

                # Mutation: v = a + F * (b - c)
                mutant = a + F * (b - c)

                # Crossover: binomial
                # Generate random numbers and random index to ensure at least one change
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Clipping to bounds
                trial = np.clip(trial, lower, upper)

                # Evaluate trial
                trial_fitness = func(trial)
                budget -= 1

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_x = trial.copy()
                        best_y = trial_fitness
                # else: keep current

        return best_x, best_y
