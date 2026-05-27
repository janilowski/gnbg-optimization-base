# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE/rand/1/bin) with adaptive dither on F and CR.
# Search state: A population of candidate solutions (size popsize) stored as a 2D numpy array.
# Candidate generation: For each target vector, three distinct individuals are selected randomly from the population (excluding the target). A mutant vector is created as: mutant = base + F * (individual2 - individual3), where base is a randomly chosen individual (rand/1 scheme). F is generated each generation from a uniform distribution U(0.5, 1.0) (dither) to improve robustness.
# Selection and replacement: Greedy selection: after binomial crossover (CR also dithering U(0.0, 1.0)) with at least one component from the mutant, the trial vector is evaluated. If its objective value is not worse than that of the target, it replaces the target in the population.
# Adaptation: No explicit parameter adaptation; F and CR are randomly sampled each generation from fixed ranges (dither). This provides a simple form of self‑adaptation across generations.
# Exploration mechanisms: Mutation with random scaling and crossover; large population size (up to 10*dim) helps maintain diversity.
# Exploitation mechanisms: Greedy replacement drives the population toward better fitness; recombination with the target vector preserves good components.
# Boundary handling: Reflection: if a coordinate falls outside [lb, ub], it is reflected back repeatedly until inside.
# Budget strategy: Initial population evaluated immediately (popsize evaluations). Remaining budget is consumed trial‑by‑trial. Generation loops stop when budget is exhausted.
# Closest known influences: Classic DE/rand/1/bin (Storn & Price, 1997) with dither on F (Price, 1997).
# Novelty or unusual aspects: Simple and compact implementation; dither on both F and CR; no control parameter tuning required.
# Failure modes: May stagnate on highly multimodal or non‑separable problems due to fixed dither ranges; poor performance on very low budgets (budget < 4 falls back to random search).
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Read bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("No bounds found in func")

        budget = self.budget
        dim = self.dim

        # Determine population size
        # At least 4 required for DE's mutation, at most 10*dim or half budget
        popsize = max(4, min(10 * dim, budget // 2))
        if popsize > budget:
            # Budget very small: fall back to random search
            best_x = lb + np.random.random(dim) * (ub - lb)
            best_y = func(best_x)
            budget -= 1
            evaluations = 1
            while budget > 0:
                x = lb + np.random.random(dim) * (ub - lb)
                y = func(x)
                evaluations += 1
                budget -= 1
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # Initialize population uniformly in [lb, ub]
        pop = lb + np.random.rand(popsize, dim) * (ub - lb)
        fitness = np.empty(popsize)
        for i in range(popsize):
            fitness[i] = func(pop[i])
        budget -= popsize

        # Track global best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main evolution loop
        while budget > 0:
            # Sample F and CR for this generation (dither)
            F = 0.5 + 0.5 * np.random.random()          # in [0.5, 1.0]
            CR = np.random.random()                     # in [0.0, 1.0]

            # Shuffle population order to process vectors randomly
            indices = np.arange(popsize)
            np.random.shuffle(indices)

            for i in indices:
                if budget <= 0:
                    break

                # Select three distinct indices different from i
                pool = [j for j in indices if j != i]
                r = np.random.choice(pool, 3, replace=False)
                base, r1, r2 = r[0], r[1], r[2]

                # Mutation: rand/1
                mutant = pop[base] + F * (pop[r1] - pop[r2])

                # Boundary handling: reflect repeatedly until inside
                for d in range(dim):
                    while mutant[d] < lb[d] or mutant[d] > ub[d]:
                        if mutant[d] < lb[d]:
                            mutant[d] = 2 * lb[d] - mutant[d]
                        elif mutant[d] > ub[d]:
                            mutant[d] = 2 * ub[d] - mutant[d]

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.empty(dim)
                for d in range(dim):
                    if np.random.random() < CR or d == j_rand:
                        trial[d] = mutant[d]
                    else:
                        trial[d] = pop[i, d]

                # Evaluate trial
                trial_y = func(trial)
                budget -= 1

                # Selection: replace if not worse
                if trial_y <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()

        return best_x, best_y
