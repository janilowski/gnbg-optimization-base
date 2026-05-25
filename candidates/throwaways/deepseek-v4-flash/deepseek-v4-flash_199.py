import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple Differential Evolution (DE) optimizer for black-box minimization.
# It uses a population of candidate solutions, mutation (rand/1/bin), binomial crossover,
# and greedy selection. It incorporates boundary clipping and dynamic population sizing
# based on evaluation budget and dimensionality.
# Search state: A population of vectors (np array of shape (pop_size, dim)), their fitness
# values (list), and the best solution found so far.
# Candidate generation: For each target vector, three distinct random individuals are
# selected from the population (excluding the target). A mutant vector is computed as
# base + F * (diff1 - diff2) with F uniformly sampled in [0.5, 1.0]. The trial vector is
# formed by binomial crossover with CR=0.9, ensuring at least one component from the mutant.
# Selection and replacement: If the trial vector's fitness is lower (better) than the
# target's, it replaces the target in the population.
# Adaptation: None; parameters F and CR are fixed.
# Exploration mechanisms: Random selection of base and difference vectors, mutation factor
# variability, crossover.
# Exploitation mechanisms: Greedy selection, best solution tracking.
# Boundary handling: Clipping each coordinate to [lb, ub].
# Budget strategy: Population size is set to max(4, min(200, budget//5, 10*dim)). The
# budget is exhausted by iterating through generations: each generation evaluates pop_size
# trial vectors, but each evaluation is checked against the budget. The algorithm halts
# when evals >= budget. The initial population consumes pop_size evaluations.
# Closest known influences: Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: Simple, no adaptive mechanisms; uses per-mutation random F
# to add diversity.
# Failure modes: May converge prematurely for multimodal functions; fixed parameters not
# optimal for all landscapes; may stagnate if population diversity is lost.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    """A simple Differential Evolution minimizer for black-box benchmarks."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ------------------------------------------------------------------
        # 1. Determine bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        # make sure they are 1-D arrays
        lb = np.ravel(lb)
        ub = np.ravel(ub)

        dim = self.dim
        budget = self.budget
        evals = 0               # evaluation counter
        best_y = np.inf
        best_x = np.empty(dim)

        # ------------------------------------------------------------------
        # 2. Determine population size
        # ------------------------------------------------------------------
        # heuristic: scale with dimension but stay within budget
        pop_size = max(4, min(200, budget // 5, 10 * dim))
        pop_size = min(pop_size, budget)          # cannot exceed budget
        # fallback to pure random search if population is too small (<4)
        if pop_size < 4:
            pop_size = budget
            # pure random search: sample until budget exhausted
            while evals < budget:
                x = np.random.uniform(lb, ub, size=dim)
                y = float(func(x))
                evals += 1
                if y < best_y:
                    best_x = x.copy()
                    best_y = y
            return best_x, best_y

        # ------------------------------------------------------------------
        # 3. Initialise population
        # ------------------------------------------------------------------
        pop = np.empty((pop_size, dim))
        fitness = np.empty(pop_size)
        for i in range(pop_size):
            if evals >= budget:
                break
            x = np.random.uniform(lb, ub, size=dim)
            y = float(func(x))
            evals += 1
            pop[i] = x
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = x.copy()

        # if we already exhausted budget, return best found
        if evals >= budget:
            return best_x, best_y

        # ------------------------------------------------------------------
        # 4. Main DE loop (rand/1/bin)
        # ------------------------------------------------------------------
        F_low, F_high = 0.5, 1.0
        CR = 0.9

        # continue generation loop until budget runs out
        while evals < budget:
            for target_idx in range(pop_size):
                if evals >= budget:
                    break

                # choose three distinct random indices different from target_idx
                indices = list(range(pop_size))
                indices.remove(target_idx)
                r = np.random.choice(indices, size=3, replace=False)
                a, b, c = r[0], r[1], r[2]

                # mutation: F sampled per mutation (scalar)
                F = np.random.uniform(F_low, F_high)
                mutant = pop[a] + F * (pop[b] - pop[c])

                # binomial crossover: at least one dimension from mutant
                trial = pop[target_idx].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # boundary clipping
                trial = np.clip(trial, lb, ub)

                # evaluate trial
                trial_y = float(func(trial))
                evals += 1

                # selection (greedy)
                if trial_y < fitness[target_idx]:
                    pop[target_idx] = trial
                    fitness[target_idx] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()

        return best_x, best_y
