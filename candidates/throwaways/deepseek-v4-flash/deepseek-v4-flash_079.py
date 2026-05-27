# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a classic Differential Evolution (DE/rand/1/bin) with a dithering mutation factor and reflection-based boundary handling. It is designed for black-box minimization on the GNBG benchmark.
# Search state: A population of candidate solutions (vectors) stored as a NumPy array of shape (pop_size, dim), along with their fitness values.
# Candidate generation: For each target vector, a mutant is created by adding the scaled difference of two randomly selected distinct population members (excluding the target) to a third random member. The scaling factor F is sampled uniformly in [0.5, 1.0] per generation.
# Selection and replacement: After binomial crossover (CR=0.9), the trial vector is evaluated. If its fitness is less than or equal to the target's fitness, the trial replaces the target in the population. The best solution ever found is tracked.
# Adaptation: F is dithered per generation (randomly chosen from a range), but no per‑individual adaptation is used to keep the implementation simple and robust.
# Exploration mechanisms: The mutation operator with random base vectors and difference vectors provides exploration across the search space. The dithering of F helps maintain diversity.
# Exploitation mechanisms: The greedy selection (replacing only if trial is better) and the binomial crossover that inherits many components from the target promote exploitation around promising solutions.
# Boundary handling: Points outside the feasible box are reflected back inside using the midpoint rule: each out‑of‑bounds coordinate is mirrored around the violated bound.
# Budget strategy: The population size is set to the nearest integer between 4 and 20 based on budget (budget//20, clamped to [4,20]). Each generation consumes pop_size function evaluations. The algorithm runs until the budget is exhausted.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997) with dithering (random F) suggested by Price et al. No additional adaptive schemes.
# Novelty or unusual aspects: None – deliberately a simple, robust reference implementation.
# Failure modes: May converge prematurely on highly multimodal problems with limited budget; the fixed CR may not suit all landscapes; reflection can cause clustering near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # population size: at least 4, at most 20, linear with budget but capped
        self.pop_size = max(4, min(20, budget // 20))
        # ensure that at least one generation is possible (budget >= pop_size)
        # but we handle early termination in __call__
        self.F_min = 0.5
        self.F_max = 1.0
        self.CR = 0.9

    def __call__(self, func):
        # ----- determine bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds: expected func.lower/upper or func.bounds.lb/ub")
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget

        # ----- initialisation -----
        # population uniformly in [lb, ub]
        pop = lb + (ub - lb) * np.random.uniform(size=(pop_size, dim))
        # evaluate initial population
        fitness = np.full(pop_size, np.inf)
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1

        best_idx = np.argmin(fitness[:evals])
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # ----- main loop -----
        while evals < budget:
            # generate a random F for this generation (dither)
            F = np.random.uniform(self.F_min, self.F_max)
            CR = self.CR  # fixed crossover rate

            # indices for the current generation (targets)
            indices = np.arange(pop_size)

            # for each target vector, generate a trial
            for target_idx in indices:
                if evals >= budget:
                    break

                # choose three distinct random indices != target_idx
                candidates = [i for i in indices if i != target_idx]
                if len(candidates) < 3:
                    break  # not enough individuals (should not happen with pop_size>=4)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # binomial crossover
                trial = pop[target_idx].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # boundary reflection
                # apply reflection for each coordinate if out of bounds
                out_low = trial < lb
                out_high = trial > ub
                # reflect: mirror around the violated bound
                trial[out_low] = 2 * lb[out_low] - trial[out_low]
                trial[out_high] = 2 * ub[out_high] - trial[out_high]
                # if still out after reflection (rare), clamp to bound
                trial = np.clip(trial, lb, ub)

                # evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # selection
                if trial_fitness <= fitness[target_idx]:
                    pop[target_idx] = trial
                    fitness[target_idx] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
