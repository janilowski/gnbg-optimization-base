# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE/rand/1/bin) minimizer with
#   boundary clipping and population size scaled to dimension and budget.
# Search state: A population of candidate vectors (numpy array of shape
#   (pop_size, dim)) plus the corresponding objective values and the current
#   best-known solution (x_best, y_best).
# Candidate generation: For each target vector, three distinct random
#   population members are selected (base, diff1, diff2). The mutant is
#   base + F * (diff1 - diff2), then binomial crossover (CR) combines the
#   mutant and the target to form the offspring.
# Selection and replacement: Greedy – the offspring replaces the target if
#   its objective value is strictly lower (minimization).
# Adaptation: No self-adaptation; F and CR are fixed (0.8 and 0.9).
# Exploration mechanisms: Mutation (random difference vectors) and crossover
#   allow wide exploration. Population diversity is maintained by replacement
#   only when an improvement is found.
# Exploitation mechanisms: The best solution is preserved and the greedy
#   replacement pushes the population toward better regions.
# Boundary handling: Offspring coordinates are clipped to the decision space
#   bounds if they fall outside.
# Budget strategy: Population size is set to min(budget, max(5*dim, 20)). The
#   initial population evaluates this many points. Then, one offspring per
#   target is generated per generation until the budget is exhausted, with
#   remaining evaluations handled sequentially (mid‑generation stop).
# Closest known influences: Classic DE/rand/1/bin (Storn & Price, 1997).
# Novelty or unusual aspects: None; implementation prioritises clarity and
#   robustness over novelty.
# Failure modes: Premature convergence on highly multimodal landscapes if
#   population diversity collapses; poor performance on extremely low‑budget
#   runs because the initial random sample may dominate.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    """
    Differential Evolution minimizer for black-box functions.
    """

    def __init__(self, budget: int, dim: int) -> None:
        self.budget = budget
        self.dim = dim

    def __call__(self, func) -> tuple[np.ndarray, float]:
        # ------------------------------------------------------------------
        # 1. Determine bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)

        # ------------------------------------------------------------------
        # 2. Population size
        # ------------------------------------------------------------------
        # At least 4 individuals are needed for DE mutation (even in 1D).
        pop_size = max(4, 5 * self.dim)
        pop_size = min(self.budget, pop_size)   # respect limited budget
        pop_size = int(pop_size)                # ensure integer

        # ------------------------------------------------------------------
        # 3. Initialisation
        # ------------------------------------------------------------------
        rng = np.random.RandomState()   # seed already set by harness
        # Uniform initial population within bounds
        population = lb + (ub - lb) * rng.rand(pop_size, self.dim)
        # Evaluate initial population
        fitness = np.full(pop_size, np.inf)
        best_y = np.inf
        best_x = np.empty(self.dim)

        remaining = self.budget
        for i in range(pop_size):
            if remaining <= 0:
                break
            fitness[i] = func(population[i])
            remaining -= 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = population[i].copy()

        # If we already exhausted the budget, return best found so far
        if remaining <= 0:
            return best_x, best_y

        # ------------------------------------------------------------------
        # 4. Differential Evolution generations
        # ------------------------------------------------------------------
        F = 0.8          # mutation factor
        CR = 0.9         # crossover probability

        # We loop until the budget runs out. Each iteration processes one
        # target vector. After a full pass through the population a generation
        # has completed, but we stop mid‑generation if budget is exhausted.
        while remaining > 0:
            for i in range(pop_size):
                if remaining <= 0:
                    break

                # --- Mutation: select three distinct random indices != i ---
                idxs = [j for j in range(pop_size) if j != i]
                if len(idxs) < 3:
                    # Should not happen because pop_size >= 4, but safety
                    break
                chosen = rng.choice(idxs, size=3, replace=False)
                base, r1, r2 = chosen[0], chosen[1], chosen[2]
                mutant = population[base] + F * (population[r1] - population[r2])

                # --- Crossover: binomial ---
                crossover_mask = rng.rand(self.dim) < CR
                # always take at least one component from the mutant
                crossover_mask[rng.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])

                # --- Boundary handling: clip to bounds ---
                trial = np.clip(trial, lb, ub)

                # --- Evaluation and selection ---
                trial_fitness = func(trial)
                remaining -= 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
