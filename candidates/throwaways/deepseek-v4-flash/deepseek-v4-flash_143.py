import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) variant with self‑adaptive parameters (jDE)
#          and stagnation‑based restart.
# Search state: Population of candidate solutions with associated fitness; each individual
#               also stores its own scaling factor F and crossover rate CR.
# Candidate generation: For each population member, a mutant is produced by the classic
#                       DE/rand/1 scheme (base + F * (difference of two other individuals),
#                       then binomial crossover with probability CR yields a trial solution.
# Selection and replacement: Greedy selection – the trial replaces the current vector if its
#                            fitness is not worse (≤). The global best is tracked.
# Adaptation: With a small probability (0.1 per generation per individual), F and CR are
#             resampled uniformly from [0.1, 1.0] and [0, 1] respectively (jDE style).
# Exploration mechanisms: High initial crossover rate (0.9) and moderate scaling factor (0.5);
#                         periodic full restart reinitialises all but the best individual uniformly,
#                         injecting new diversity when stagnation is detected.
# Exploitation mechanisms: Greedy replacement, mutation based on current population
#                          differences, and the preservation of successful parameters via adaptation.
# Boundary handling: Trial solutions are clipped component‑wise to the problem bounds.
# Budget strategy: Population size is chosen adaptively based on budget and dimension.
#                  Every function evaluation is counted precisely; a restart is triggered only
#                  when enough remaining budget exists to reinitialise the entire population.
# Closest known influences: jDE (Brest et al., 2006) without archive or rank‑based adaptation.
# Novelty or unusual aspects: Minimalistic implementation of jDE with a simple restart
#                             mechanism that discards all but the best solution.
# Failure modes: May stagnate on highly non‑separable high‑dimensional problems when population
#                size is too small; restart can waste evaluations if budget is tight.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ----- Determine bounds -----
        try:
            lower = func.lower
            upper = func.upper
        except AttributeError:
            lower = func.bounds.lb
            upper = func.bounds.ub
        lb = np.array(lower, dtype=float)
        ub = np.array(upper, dtype=float)

        dim = self.dim
        budget = self.budget

        # ----- Population size (scaled to budget and dimension) -----
        pop_size = max(4, min(100, int(budget / 5), dim * 5))
        pop_size = min(pop_size, budget)          # avoid larger than budget

        # ----- Initialisation -----
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_x = np.empty(dim)
        best_y = np.inf
        evals = 0

        # jDE parameters (one per individual)
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)

        # Evaluate initial population
        for i in range(pop_size):
            y = func(pop[i])
            evals += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # ----- Main evolution loop -----
        stagnation_limit = max(5, dim)
        stagnate = 0

        while evals < budget:
            improved = False
            for i in range(pop_size):
                # ---- Mutation (DE/rand/1) ----
                # Choose three distinct random indices ≠ i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # ---- jDE parameter adaptation ----
                if np.random.rand() < 0.1:
                    F[i] = np.random.uniform(0.1, 1.0)
                if np.random.rand() < 0.1:
                    CR[i] = np.random.uniform(0.0, 1.0)

                # ---- Mutant and crossover ----
                mutant = pop[r1] + F[i] * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)          # boundary clipping

                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]

                # ---- Evaluation ----
                if evals >= budget:
                    break
                trial_fitness = func(trial)
                evals += 1

                # ---- Selection ----
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()
                        improved = True

            # ---- Stagnation restart ----
            if improved:
                stagnate = 0
            else:
                stagnate += 1

            # Restart only if enough budget remains to reinitialise all but the best
            if stagnate >= stagnation_limit and (evals + pop_size - 1 <= budget):
                # Keep best individual at index 0
                pop[0] = best_x
                fitness[0] = best_y
                # Reinitialise the rest uniformly
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    pop[i] = np.random.uniform(lb, ub)
                    fitness[i] = func(pop[i])
                    evals += 1
                    if fitness[i] < best_y:
                        best_y = fitness[i]
                        best_x = pop[i].copy()
                # Reset jDE parameters
                F[:] = 0.5
                CR[:] = 0.9
                stagnate = 0

        return best_x, best_y
