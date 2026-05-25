# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm implements a standard Differential Evolution (DE) optimizer for black-box minimization.
# Search state: A population of candidate solution vectors and their corresponding fitness values.
# Candidate generation: For each target vector, a mutant vector is created using the DE/rand/1 strategy (base + F * (donor1 - donor2)). Binomial crossover then combines target and mutant to form a trial vector.
# Selection and replacement: Greedy selection: the trial replaces the target if it yields a lower (better) objective value. No explicit elitism (best solution is tracked separately).
# Adaptation: Fixed DE parameters: scaling factor F = 0.5, crossover probability CR = 0.9. Population size is set adaptively as min(200, 10*dim), with a minimum of 4.
# Exploration mechanisms: Random mutation based on differences between population members ensures global exploration, especially early in the run.
# Exploitation mechanisms: As the population converges, difference vectors shrink, focusing search around promising regions. Greedy selection drives local refinement.
# Boundary handling: Trial vectors that violate bounds are clipped to the nearest bound (truncation).
# Budget strategy: The initial population is evaluated, followed by successive generations until the budget is exhausted. If insufficient budget remains for a full generation, only the necessary number of trial vectors are generated.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997), specifically the DE/rand/1/bin variant.
# Novelty or unusual aspects: None; the implementation is a straightforward textbook DE with minor budget management logic.
# Failure modes: May stall on highly multimodal landscapes if diversity drops prematurely. Fixed parameters may underperform on some function classes. Clipping can cause stagnation near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self._popsize = max(4, min(200, 10 * dim))

    def __call__(self, func):
        """Minimize func subject to the evaluation budget.

        Returns (best_x, best_y) where best_y = func(best_x) is minimal.
        """
        # ----- read bounds from the callable -------------------------------------------------
        try:
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        dim = self.dim
        popsize = self._popsize

        # ----- initialise population ---------------------------------------------------------
        pop = np.random.uniform(lower, upper, (popsize, dim))
        fits = np.array([func(x) for x in pop])
        evals = popsize

        # track best
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_y = fits[best_idx]

        # ----- DE parameters (fixed) ---------------------------------------------------------
        F = 0.5
        CR = 0.9

        # ----- main loop (generations) -------------------------------------------------------
        while evals < self.budget:
            # determine how many trials we can still perform this generation
            remaining = self.budget - evals
            trials_this_gen = min(popsize, remaining)

            for i in range(trials_this_gen):
                # pick three distinct random indices (i is excluded from the pool)
                indices = list(range(popsize))
                indices.remove(i)
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)

                # mutation: DE/rand/1
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # binomial crossover
                mask = np.random.rand(dim) < CR
                # ensure at least one component comes from mutant
                jrand = np.random.randint(dim)
                mask[jrand] = True
                trial = np.where(mask, mutant, pop[i])

                # boundary handling: clip to [lower, upper]
                trial = np.clip(trial, lower, upper)

                # evaluate trial
                trial_fit = func(trial)
                evals += 1

                # greedy selection
                if trial_fit < fits[i]:
                    pop[i] = trial
                    fits[i] = trial_fit
                    if trial_fit < best_y:
                        best_x = trial.copy()
                        best_y = trial_fit

                # after each evaluation check if budget exhausted (exit loop early)
                if evals >= self.budget:
                    break

        return best_x, best_y
