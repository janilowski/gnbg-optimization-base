# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Simple differential evolution (DE/rand/1/bin) with fixed parameters.
# Search state: population of candidate solutions and their fitness.
# Candidate generation: mutation and crossover to create trial vectors.
# Selection and replacement: greedy selection; trial replaces parent if better.
# Adaptation: none; fixed F=0.8, CR=0.9.
# Exploration: mutation using random differential vectors; uniform crossover.
# Exploitation: selection pressure from greedy replacement.
# Boundary handling: clip to bounds after mutation.
# Budget strategy: limit population size to allow many generations; break when budget exhausted.
# Closest known influences: Standard differential evolution (Storn & Price, 1997).
# Novelty or unusual aspects: none.
# Failure modes: may stagnate on highly multimodal or deceptive functions; fixed parameters may not suit all problems.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the algorithm with a function evaluation budget and problem dimension.

        Args:
            budget: maximum number of function evaluations allowed
            dim: dimensionality of the search space
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the black-box minimisation algorithm on the provided function.

        Args:
            func: an objective function object providing .lower/.upper or .bounds.lb/.bounds.ub

        Returns:
            (best_x, best_y): best found solution and its value
        """
        # --- read bounds from the function object ---
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb, dtype=float)
                ub = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Cannot read bounds from the provided function object")
        dim = self.dim

        # --- population size: scale with dimension but stay within budget ---
        max_pop = min(100, self.budget - 1)               # keep at least one eval for later
        pop_size = max(4, min(10 * dim, max_pop // 2, 100))

        # --- initialise population uniformly in the domain ---
        pop = lb + np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb)
        fitness = np.full(pop_size, np.inf)

        evals = 0
        best_x = None
        best_y = np.inf

        # --- evaluate initial population ---
        for i in range(pop_size):
            y = func(pop[i])
            evals += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # --- DE parameters (fixed) ---
        F = 0.8
        CR = 0.9

        # --- main evolutionary loop ---
        while evals < self.budget:
            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # pick three distinct random indices different from i
                indices = [j for j in range(pop_size) if j != i]
                r = np.random.choice(indices, size=3, replace=False)
                a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]

                # mutation
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, lb, ub)   # boundary handling

                # binomial crossover
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # evaluate trial
                f_trial = func(trial)
                evals += 1

                # greedy selection
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()

        return best_x, best_y
