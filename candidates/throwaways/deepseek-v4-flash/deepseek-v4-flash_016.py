# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact differential evolution (DE/rand/1/bin) minimizer.    
# Search state: A population of candidate vectors with their objective values.
# Candidate generation: Each target vector is perturbed by the scaled difference
#   of two other random population members (mutation) and then recombined with
#   the target via binomial crossover.
# Selection and replacement: Greedy – a trial vector replaces the target if it
#   yields a strictly better (lower) objective value.
# Adaptation: Fixed differential weight F=0.8 and crossover probability CR=0.9;
#   no self‑adaptation.
# Exploration mechanisms: The differential mutation drives global search;
#   high CR encourages mixing of dimensions.
# Exploitation mechanisms: Population elitism is implicit (best individual
#   remains in the population without explicit archiving).
# Boundary handling: Trial vectors are clipped componentwise to the box bounds.
# Budget strategy: Evaluations are counted; the loop stops as soon as the
#   budget is exhausted or no evaluations remain.
# Closest known influences: Standard DE/rand/1/bin as described by Storn & Price.
# Novelty or unusual aspects: None; the implementation is plain and robust.
# Failure modes: May converge slowly on highly multimodal or non‑separable
#   landscapes; fixed parameters may be suboptimal for some functions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # read bounds
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        # ensure arrays of correct dimension
        lb = np.broadcast_to(lb, (self.dim,)).copy()
        ub = np.broadcast_to(ub, (self.dim,)).copy()

        # population size – heuristic: at least 10, at most budget/2
        pop_size = min(self.budget // 2, max(10, 4 * self.dim))
        # ensure at least 4 individuals (needed for mutation)
        pop_size = max(pop_size, 4)

        # initialisation
        pop = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            fitness[i] = func(pop[i])
            evals += 1

        # keep best of the initial population
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # DE parameters
        F = 0.8
        CR = 0.9

        # main loop
        while evals < self.budget:
            # generate one trial per target
            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # mutation: pick three distinct random indices ≠ i
                idxs = [idx for idx in range(pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(idxs, size=3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # crossover: binomial
                crossover_mask = np.random.rand(self.dim) < CR
                # ensure at least one dimension from mutant
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(self.dim)] = True
                trial = np.where(crossover_mask, mutant, pop[i])

                # boundary clamping
                trial = np.clip(trial, lb, ub)

                # evaluation
                trial_fitness = func(trial)
                evals += 1

                # greedy selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
