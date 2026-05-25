import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) optimizer for black‑box minimization.
#          Uses the classic DE/rand/1/bin strategy with fixed parameters.
# Search state: A population of NP candidate vectors with their fitness values.
# Candidate generation: For each parent, a mutant vector is formed by adding
#          F * (random1 - random2) to a third random vector. Binomial crossover
#          mixes the mutant with the parent with probability CR.
# Selection and replacement: Greedy – a trial replaces its parent if it has
#          lower (better) objective value.
# Adaptation: None – F and CR are fixed at 0.8 and 0.9 respectively.
# Exploration mechanisms: High CR (0.9) and moderate F (0.8) maintain diversity;
#          mutation uses three distinct random individuals.
# Exploitation mechanisms: The population contracts around good solutions through
#          selection, and the mutation step length scales with population spread.
# Boundary handling: Trial points are clipped to the decision variable bounds.
# Budget strategy: The entire evaluation budget is consumed. The initial
#          population uses NP evaluations, then the main loop generates exactly
#          one trial per parent per generation until the budget is exhausted.
# Closest known influences: Storn & Price (1995) Differential Evolution – DE/rand/1/bin.
# Novelty or unusual aspects: None; a straightforward, standard DE implementation.
# Failure modes: Fixed parameters may not suit all landscapes; on extremely
#          noisy or high‑dimensional problems the algorithm may stagnate.
#          Very low budgets (< 4 evaluations) force a fallback to random search.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """Prepare the optimizer for a black‑box minimization run."""
        self.budget = budget
        self.dim = dim
        # DE parameters – can be tuned; here kept simple.
        self.F = 0.8          # mutation factor
        self.CR = 0.9         # crossover probability
        # Population size: at least 4, at most budget, scales mildly with dim.
        self.NP = min(budget, max(4, 4 * dim))
        # Fallback: if budget is too small for one DE generation, use random search.
        self._use_random_fallback = (self.NP < 4)

    def __call__(self, func):
        """Minimize `func` and return (best_x, best_y)."""
        # ---- read bounds ----------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = func.bounds.lb
            ub = func.bounds.ub
            lower = np.asarray(lb, dtype=float)
            upper = np.asarray(ub, dtype=float)
        else:
            raise AttributeError("Objective must provide lower/upper or bounds.lb/ub")

        dim = self.dim
        budget = self.budget
        evals = 0

        # ---- fallback for extremely small budgets ---------------------------
        if self._use_random_fallback:
            best_x = None
            best_y = np.inf
            while evals < budget:
                x = np.random.uniform(lower, upper, size=dim)
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # ---- population initialisation --------------------------------------
        NP = self.NP
        pop = np.random.uniform(lower, upper, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        for i in range(NP):
            fitness[i] = func(pop[i])
            evals += 1
            if evals >= budget:
                # budget exhausted after initial evaluation, return best so far
                best_idx = np.argmin(fitness[:i+1])
                return pop[best_idx].copy(), fitness[best_idx]

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # ---- main DE loop ---------------------------------------------------
        while evals < budget:
            for i in range(NP):
                # if we are about to run out, do one more trial per remaining budget
                if evals >= budget:
                    break

                # mutation: pick three distinct indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])

                # binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # boundary clipping
                trial = np.clip(trial, lower, upper)

                # evaluate
                ftrial = func(trial)
                evals += 1

                # selection
                if ftrial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = ftrial
                    if ftrial < best_y:
                        best_y = ftrial
                        best_x = trial.copy()

                # update global best (handled in selection if better)
            # optional: after a generation, we could adapt parameters – not done here

        return best_x, best_y
