# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a standard Differential Evolution (DE/rand/1/bin) minimizer.
#          It is a population-based evolutionary algorithm suitable for continuous black‑box
#          optimization. The implementation is compact, relies only on NumPy, and respects the
#          evaluation budget.
# Search state: A matrix ‘pop’ of shape (NP, dim) holds all candidate solutions, and a vector
#               ‘fitness’ stores their objective values. The current best individual and its
#               fitness are tracked separately.
# Candidate generation: At each generation, for each population member i, a mutant vector is
#                       created using the “rand/1” scheme: v = r1 + F * (r2 - r3), where r1, r2, r3
#                       are distinct random indices different from i. A trial point is then formed
#                       via binomial crossover with the parent i, controlled by crossover rate CR.
# Selection and replacement: Selection is deterministic greedy. If the trial point has a lower
#                            (better) objective value than the parent, it replaces the parent in the
#                            population.
# Adaptation: No hyperparameters are adapted during the run. The mutation factor F = 0.8 and
#             crossover rate CR = 0.9 remain fixed.
# Exploration mechanisms: Diversity is maintained through the stochastic nature of mutation
#                         (differences between random population members) and crossover. The
#                         population itself spreads across the search space.
# Exploitation mechanisms: Exploitation is achieved via the differential mutation that tends to
#                          produce small perturbations when the population converges, and through
#                          the greedy replacement that retains only improving points.
# Boundary handling: Clipping (projection) is applied: any coordinate of the trial vector that
#                    falls outside [lb, ub] is set to the nearest bound.
# Budget strategy: The algorithm first evaluates an initial random population. Then it runs the
#                  DE loop generation by generation, each generation costing NP evaluations.
#                  The loop stops when the next generation would exceed the remaining budget. If
#                  the budget is too small even for a full generation, only the initial
#                  evaluations are performed.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997) with the
#                           “rand/1/bin” strategy.
# Novelty or unusual aspects: None. This is a straightforward, unmodified DE implementation.
# Failure modes: May converge prematurely to a local optimum on highly multimodal landscapes;
#                static hyperparameters are not optimal for all problems; on very low budgets
#                it reverts to random search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # Differential Evolution parameters (fixed)
        self.F = 0.8        # mutation factor
        self.CR = 0.9       # crossover rate

        # Population size: aim for ~4*dim but respect budget
        self.NP = max(4, min(budget, 4 * dim))

        # If the budget is very small, use a population of 2 anyway
        if self.NP < 4 and budget >= 4:
            self.NP = 4

    def __call__(self, func):
        """Run the optimizer for the given black‑box function func.

        func must provide bounds via either:
            - func.lower, func.upper   (both numpy arrays or scalars)
            - func.bounds.lb, func.bounds.ub   (both numpy arrays or scalars)
        Returns (best_x, best_y) where best_y is the minimum objective found.
        """
        # ---- 1. extract bounds ----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: assume standard bounds (should not happen in the benchmark)
            lb = np.full(self.dim, -10.0)
            ub = np.full(self.dim, 10.0)

        # Ensure arrays
        lb = np.broadcast_to(lb, (self.dim,))
        ub = np.broadcast_to(ub, (self.dim,))

        # ---- 2. initialization ----
        NP = self.NP
        dim = self.dim
        budget = self.budget
        evals = 0

        # helper: evaluate and count
        def evaluate(x):
            nonlocal evals
            evals += 1
            return func(x)

        # population
        pop = lb + (ub - lb) * np.random.uniform(0, 1, (NP, dim))
        fitness = np.full(NP, np.inf)
        best_x = np.copy(pop[0])
        best_y = np.inf

        for i in range(NP):
            y = evaluate(pop[i])
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = np.copy(pop[i])
            if evals >= budget:
                # budget already exhausted after initialization (unlikely)
                return best_x, best_y

        # ---- 3. main DE loop ----
        # Each generation processes all NP individuals, costing NP evaluations.
        # We stop as soon as the next generation would exceed the budget.
        while evals + NP <= budget:
            for i in range(NP):
                # choose three distinct random indices different from i
                idxs = list(range(NP))
                idxs.remove(i)
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

                # mutation
                v = pop[r1] + self.F * (pop[r2] - pop[r3])

                # binomial crossover
                trial = np.where(
                    np.random.rand(dim) < self.CR,
                    v,
                    pop[i]
                )
                # ensure at least one coordinate from mutant (standard DE)
                j_rand = np.random.randint(dim)
                trial[j_rand] = v[j_rand]

                # boundary handling: clip
                trial = np.clip(trial, lb, ub)

                # evaluate trial
                y_trial = evaluate(trial)

                # selection
                if y_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = y_trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = np.copy(trial)

                # budget check inside loop (should not happen because we guaranteed NP evaluations)
                if evals >= budget:
                    return best_x, best_y

        # final return
        return best_x, best_y
