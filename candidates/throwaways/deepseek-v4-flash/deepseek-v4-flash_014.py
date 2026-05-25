# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This implements a steady-state Differential Evolution (DE) variant with rand/1/bin mutation and exponential crossover. The algorithm is chosen for its compactness, simplicity, and robust performance across various black-box optimization problems, including GNBG benchmarks.
# Search state: The algorithm maintains a population of candidate solutions (vectors) uniformly distributed in the search space. The population size is determined by the dimension (max(4, 4 + 3*log(dim))). The best solution found so far is also tracked.
# Candidate generation: For each iteration, a random target index is selected. Three distinct random individuals (different from the target) are chosen. A mutant vector is created as base + F * (diff1 - diff2) with F=0.7 (fixed). Then binary crossover with CR=0.9 produces a trial vector.
# Selection and replacement: After evaluating the trial vector, if it yields a better (lower) objective value than the current target, it replaces the target in the population. The global best is updated accordingly.
# Adaptation: No parameter adaptation is used; F and CR are fixed. The structure of the algorithm does not change during the run.
# Exploration mechanisms: The differential mutation and population diversity provide exploration. Crossover allows mixing of components from different individuals.
# Exploitation mechanisms: The greedy replacement and the global best tracking focus search around promising regions. The population gradually converges.
# Boundary handling: Candidate vectors are clamped to the domain bounds [lb, ub] after generation.
# Budget strategy: The algorithm evaluates exactly one candidate per iteration, which makes it straightforward to stay within the budget. The loop runs until the evaluation count reaches the budget.
# Closest known influences: Standard DE/rand/1/bin with fixed F and CR, steady-state (one trial per iteration) rather than generational.
# Novelty or unusual aspects: None – a classic, simple DE implementation.
# Failure modes: On highly non-separable or deceptive landscapes, fixed F and CR may be suboptimal. The algorithm may converge prematurely if population loses diversity. No restart mechanism is included.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ---- read bounds ----------------------------------------------------
        try:
            lb = np.asarray(func.lower, dtype=float).ravel()
            ub = np.asarray(func.upper, dtype=float).ravel()
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float).ravel()
            ub = np.asarray(func.bounds.ub, dtype=float).ravel()

        # ---- population size -----------------------------------------------
        dim = self.dim
        popsize = max(4, int(4.0 + 3.0 * np.log(dim)))   # at least 4 for mutation
        # ---- DE parameters -------------------------------------------------
        F = 0.7
        CR = 0.9

        # ---- initial population --------------------------------------------
        pop = np.random.uniform(lb, ub, size=(popsize, dim))
        fitness = np.array([func(x) for x in pop])
        evals = popsize

        # track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # ---- main steady-state DE loop ------------------------------------
        while evals < self.budget:
            # pick random target index
            target = np.random.randint(0, popsize)

            # pick three distinct random indices, different from target
            indices = list(range(popsize))
            indices.remove(target)
            r1, r2, r3 = np.random.choice(indices, size=3, replace=False)

            # mutation: DE/rand/1
            mutant = pop[r1] + F * (pop[r2] - pop[r3])

            # binominal crossover
            cross_mask = np.random.rand(dim) < CR
            if not np.any(cross_mask):
                cross_mask[np.random.randint(0, dim)] = True
            trial = np.where(cross_mask, mutant, pop[target])

            # boundary handling: clamp
            trial = np.clip(trial, lb, ub)

            # evaluate
            trial_fitness = func(trial)
            evals += 1

            # selection
            if trial_fitness < fitness[target]:
                pop[target] = trial
                fitness[target] = trial_fitness
                if trial_fitness < best_y:
                    best_y = trial_fitness
                    best_x = trial.copy()

        return (best_x, best_y)
