import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact differential evolution (DE/rand/1/bin) minimizer, suitable
#          for the GNBG black-box benchmark.  It uses a fixed population size
#          derived from the problem dimension, and a simple clipping boundary
#          handler.
# Search state: The population (array of candidate solutions) and their fitness
#               values.  The best-known solution (best_x, best_y) is tracked
#               and returned.
# Candidate generation: Standard DE mutation (rand/1) with a scaling factor F
#                       and binomial crossover with rate CR.  Three distinct
#                       random population members are chosen for each target.
# Selection and replacement: Greedy selection: trial replaces target if its
#                            fitness is lower (minimization) or equal (to
#                            promote diversity).
# Adaptation: None – the parameters F and CR are fixed.
# Exploration mechanisms: High CR and moderate F encourage spread; random
#                         selection of the base vector and two difference
#                         vectors ensures exploration.
# Exploitation mechanisms: The greedy replacement keeps improving solutions,
#                          and the population gradually contracts around
#                          promising regions.
# Boundary handling: Trial vectors are clipped to the search bounds.
# Budget strategy: The algorithm stops when the total number of objective
#                  evaluations reaches the budget.  The population is
#                  evaluated once initially, then one new trial per
#                  generation per population member.
# Closest known influences: Classic differential evolution (Storn & Price, 1997).
# Novelty or unusual aspects: Very minimal – fixed parameters, no adaptive
#                             mechanisms, no restarts.
# Failure modes: May stagnate on highly multimodal landscapes or when the
#                problem dimension is very high relative to the budget.
#                Fixed parameters may not suit all functions.  No explicit
#                mechanism for escaping local optima.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ------------------------------------------------------------------
        # 1. Read bounds
        # ------------------------------------------------------------------
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            # fallback to func.bounds
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        # Ensure they are 1-d arrays of length dim
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)

        # ------------------------------------------------------------------
        # 2. Algorithm parameters (fixed)
        # ------------------------------------------------------------------
        # Population size: trade-off between diversity and budget consumption
        npop = max(4 * self.dim, 30)
        # Never exceed budget; one initial evaluation per member, then each
        # generation consumes popsize evaluations.
        npop = min(npop, self.budget // 2)          # at least 2 generations
        if npop < 4:
            npop = min(4, self.budget)              # fallback for tiny budget
        F = 0.8       # scaling factor
        CR = 0.9      # crossover probability

        # ------------------------------------------------------------------
        # 3. Initialisation
        # ------------------------------------------------------------------
        pop = lb + np.random.rand(npop, self.dim) * (ub - lb)
        fitness = np.empty(npop)
        best_x = None
        best_y = float('inf')
        evals = 0

        for i in range(npop):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()

        # ------------------------------------------------------------------
        # 4. Main generation loop
        # ------------------------------------------------------------------
        generation = 0
        while evals < self.budget:
            for i in range(npop):
                # --- mutation (DE/rand/1) ---
                # pick three distinct random indices different from i
                candidates = [j for j in range(npop) if j != i]
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])

                # --- binomial crossover ---
                trial = pop[i].copy()
                # ensure at least one dimension is crossed over
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # --- boundary clipping ---
                trial = np.clip(trial, lb, ub)

                # --- evaluation and selection ---
                trial_fit = func(trial)
                evals += 1
                if trial_fit <= fitness[i]:   # <= to allow neutral moves
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

                # Check budget after each evaluation
                if evals >= self.budget:
                    break

            generation += 1

        return best_x, best_y
