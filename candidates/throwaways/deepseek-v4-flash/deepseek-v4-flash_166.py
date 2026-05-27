import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a standard Differential Evolution algorithm (DE/rand/1/bin) for continuous black-box minimization.
#          Uses a population of candidate solutions, mutation with a fixed differential weight F, binomial crossover with rate CR,
#          and greedy selection. Boundary handling via simple clipping. The algorithm runs as many complete generations as the budget allows.
# Search state: A population of vectors (pop) and their corresponding function values (fitness). Also maintains the best solution found (x_best, y_best).
# Candidate generation: For each index i in the population, three distinct random indices a,b,c (all different from i) are chosen.
#          The mutant vector is computed as pop[a] + F * (pop[b] - pop[c]). Then a trial vector is formed by binomial crossover with pop[i].
# Selection and replacement: If the trial vector yields a lower (better) objective value than pop[i], it replaces pop[i].
# Adaptation: No adaptive parameters; F and CR are fixed.
# Exploration mechanisms: Mutation using scaled differences between random population members provides exploration. Crossover maintains diversity.
# Exploitation mechanisms: Greedy selection gradually focuses the population around better regions. As the population converges, mutation steps shrink.
# Boundary handling: Trial components are clipped to the lower and upper bounds.
# Budget strategy: The initial population is evaluated, then full generations (each consuming pop_size evaluations) are run while budget remains.
#          Leftover budget after the last complete generation is unused.
# Closest known influences: Classic DE/rand/1/bin (Storn & Price, 1997).
# Novelty or unusual aspects: None; straightforward textbook implementation.
# Failure modes: May converge prematurely on multimodal landscapes; clipping can cause stagnation near boundaries; fixed population size may be suboptimal for very low/high budget.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # --- read bounds ---
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        d = self.dim
        # Ensure lb/ub are 1D arrays of length d
        if lb.ndim == 0:
            lb = np.full(d, lb)
            ub = np.full(d, ub)

        # --- algorithm parameters ---
        F = 0.8          # mutation factor
        CR = 0.9         # crossover rate

        # Population size: at least 3 (needed for mutation) and at most budget//2 to allow some generations
        pop_size = min(10 * d, self.budget // 2)
        pop_size = max(pop_size, 3, 5)  # ensure reasonable minimum

        # But if budget is too small to even initialize pop_size, reduce further.
        pop_size = min(pop_size, self.budget)

        # --- initialisation ---
        pop = np.random.uniform(lb, ub, size=(pop_size, d))
        fitness = np.full(pop_size, np.inf)

        evals = 0
        best_x = None
        best_y = np.inf

        # evaluate initial population
        for i in range(pop_size):
            y = func(pop[i])
            evals += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # --- main DE loop ---
        while evals + pop_size <= self.budget:
            for i in range(pop_size):
                # select three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                # mutation
                mutant = pop[a] + F * (pop[b] - pop[c])

                # crossover
                j_rand = np.random.randint(d)
                trial = np.empty(d)
                for j in range(d):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i, j]

                # boundary handling: clip
                trial = np.clip(trial, lb, ub)

                # evaluation
                y_trial = func(trial)
                evals += 1

                # selection
                if y_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = y_trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()

                # check remaining budget after each evaluation?
                # We already check before entering generation, but to be safe:
                if evals >= self.budget:
                    break

            if evals >= self.budget:
                break

        # If budget allows one more evaluation (e.g., leftover), we could optionally
        # do a random local perturbation around best, but we keep it simple.
        return best_x, best_y
