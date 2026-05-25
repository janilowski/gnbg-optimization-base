import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact implementation of Differential Evolution (DE/rand/1/bin) for
# black-box minimization on continuous domains. It is designed to be robust
# across dimensions and respects a given evaluation budget.
# Search state: A population of candidate vectors (current generation) and their
# objective values. The best-so-far solution and its value are tracked.
# Candidate generation: For each target vector, a mutant vector is created by
# adding the scaled difference between two random population vectors to a third
# random vector (rand/1). Then binomial crossover combines the mutant with the
# target to form a trial vector.
# Selection and replacement: (μ+λ) like greedy selection: trial replaces target
# if it is not worse (less or equal to). This ensures monotonic improvement in the
# population's best.
# Adaptation: Fixed control parameters (F=0.8, Cr=0.9, NP=10*dim (min 10)).
# Exploration mechanisms: Random differential mutation and crossover spread
# trial vectors across the search space; the population-based search encourages
# diversity.
# Exploitation mechanisms: Greedy selection favors better solutions; as the
# population converges, mutation step sizes shrink implicitly because differences
# between vectors become smaller.
# Boundary handling: After crossover, each coordinate of the trial vector is
# clipped to the lower/upper bounds.
# Budget strategy: The algorithm terminates immediately when the number of
# function evaluations reaches the allotted budget; no extra evaluations are
# wasted.
# Closest known influences: Standard differential evolution (Storn & Price,
# 1997), specifically DE/rand/1/bin.
# Novelty or unusual aspects: None. Straightforward implementation; boundary
# clipping is the only deviation from the classic.
# Failure modes: May struggle on highly multimodal or separable functions if the
# population size is too small (by default scaled with dimension); fixed control
# parameters may not suit all functions. The simple clipping can cause
# stagnation near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Differential Evolution (DE/rand/1/bin) for black-box minimization."""
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # Population size: at least 10, scaled with dimension
        self.NP = max(10, 10 * dim)  # rule of thumb: 10*dim
        self.F = 0.8   # mutation scale factor
        self.Cr = 0.9  # crossover probability

    def __call__(self, func):
        """Minimize func within the evaluation budget.
        Returns (best_x, best_y) where best_y = min over all evaluations.
        """
        # ---- read bounds ----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds; func must have lower/upper or bounds.lb/bounds.ub")
        # ensure they are 1-d arrays
        lower = lower.ravel()
        upper = upper.ravel()
        dim = self.dim
        # adjust dimensions
        if len(lower) != dim:
            lower = np.full(dim, lower[0])
        if len(upper) != dim:
            upper = np.full(dim, upper[0])

        # ---- initialisation ----
        NP = self.NP
        # random uniform between lower and upper
        population = np.random.uniform(low=lower, high=upper, size=(NP, dim))
        # evaluate entire population (cost NP evaluations)
        fitness = np.array([func(x) for x in population])
        evals = NP
        # track best
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]

        # ---- main DE loop ----
        while evals + 1 <= self.budget:  # allow at least one more trial (each trial costs 1 eval)
            # create new generation (in place, but we need trial vectors)
            for i in range(NP):
                # pick three distinct random indices different from i
                idxs = list(range(NP))
                idxs.remove(i)
                a, b, c = np.random.choice(idxs, size=3, replace=False)
                # mutation: v = r1 + F*(r2 - r3)
                mutant = population[a] + self.F * (population[b] - population[c])
                # crossover: binomial, create trial
                trial = population[i].copy()
                # ensure at least one component from mutant
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < self.Cr or j == j_rand:
                        trial[j] = mutant[j]
                # boundary clipping
                trial = np.clip(trial, lower, upper)

                # evaluate trial (only if budget allows)
                if evals >= self.budget:
                    break
                trial_fit = func(trial)
                evals += 1

                # selection: replace if not worse
                if trial_fit <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit
                    # update global best
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

            # Early exit if budget exhausted inside inner loop
            if evals >= self.budget:
                break

        return best_x, best_y
