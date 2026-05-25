import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A steady-state Differential Evolution (DE) minimizer with dithering
#          and binomial crossover.  Designed to be robust across dimensions
#          and budgets while using only numpy and the standard library.
# Search state: A population of candidate solutions (array shape (npop, dim))
#               and their corresponding objective values (array shape (npop,)).
#               The global best solution and its value are tracked separately.
# Candidate generation: For each parent, a mutant is created using the
#                       DE/rand/1 scheme with a dithering scale factor F
#                       uniformly sampled in [0.5, 1.0] at each trial.
#                       The mutant is then crossed over with the parent using
#                       binomial crossover (CR = 0.9) to produce the trial.
# Selection and replacement: Greedy selection: if the trial is not worse than
#                            the current parent (fitness <= parent), it
#                            replaces the parent in the population immediately
#                            (steady-state update).
# Adaptation: Only the mutation scale factor F is adapted through dithering;
#             no sophisticated parameter control is used.
# Exploration mechanisms: High crossover rate (0.9) and dithering F promote
#                         diversity; random initialisation covers the whole
#                         bounded domain.
# Exploitation mechanisms: The greedy replacement and the storage of the
#                          overall best solution allow the algorithm to
#                          concentrate on promising regions over generations.
# Boundary handling: Candidate solutions are clipped to the box bounds.
# Budget strategy: The population size is chosen dynamically:
#                  npop = max(4, min(budget//2, 30)).  This ensures at least
#                  one full generation if the budget is sufficient, and never
#                  allocates more than half the budget to the initial
#                  population, leaving room for iterations.
# Closest known influences: Classic DE/rand/1/bin with dithering (Price,
#                           Storn, Lampinen).  Steady-state variant.
# Novelty or unusual aspects: Minimal – straightforward application of a
#                             standard algorithm with a conservative budget‑
#                             aware population sizing.
# Failure modes: May converge prematurely on highly multimodal landscapes
#                with limited budget; fixed CR might be suboptimal for some
#                problems; clamping at boundaries can reduce population
#                diversity when the optimum lies exactly on the boundary.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """
    Steady-state Differential Evolution minimizer.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # --- bounds --------------------------------------------------------
        if hasattr(func, "lower"):
            lb = np.asarray(func.lower, dtype=np.float64)
            ub = np.asarray(func.upper, dtype=np.float64)
        else:                       # assume func.bounds with lb/ub
            lb = np.asarray(func.bounds.lb, dtype=np.float64)
            ub = np.asarray(func.bounds.ub, dtype=np.float64)

        # --- population size ------------------------------------------------
        npop = max(4, min(self.budget // 2, 30))

        # --- initialisation ------------------------------------------------
        pop = lb + (ub - lb) * np.random.rand(npop, self.dim)
        fit = np.empty(npop)

        evaluations = 0
        best_x = np.empty(self.dim)
        best_y = np.inf

        for i in range(npop):
            y = func(pop[i])
            fit[i] = y
            evaluations += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        if evaluations >= self.budget:
            return best_x, best_y

        # --- main loop ----------------------------------------------------
        while evaluations < self.budget:
            for i in range(npop):
                if evaluations >= self.budget:
                    break

                # pick three distinct random indices different from i
                idxs = list(range(npop))
                idxs.remove(i)
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

                # dithering mutation factor
                F = 0.5 + 0.5 * np.random.rand()

                # mutation: DE/rand/1
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)

                # binomial crossover
                cr = 0.9
                j_rand = np.random.randint(self.dim)
                mask = np.random.rand(self.dim) < cr
                mask[j_rand] = True        # ensure at least one coordinate
                trial = np.where(mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                # evaluate trial
                ftrial = func(trial)
                evaluations += 1

                # update global best
                if ftrial < best_y:
                    best_y = ftrial
                    best_x = trial.copy()

                # greedy replacement (steady-state)
                if ftrial <= fit[i]:
                    pop[i] = trial
                    fit[i] = ftrial

        return best_x, best_y
