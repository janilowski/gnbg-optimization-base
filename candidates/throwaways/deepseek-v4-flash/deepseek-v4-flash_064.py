import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE/rand/1/bin) for black-box minimization.
# Search state: A population of NP candidate solutions sorted by fitness (implicitly).
# Candidate generation: For each target vector, a mutant is created by adding the scaled
#   difference of two other randomly selected population vectors (rand/1 scheme).
# Selection and replacement: Greedy – if the trial vector is not worse than the target,
#   it replaces the target in the population.
# Adaptation: None; mutation factor F and crossover rate CR are fixed (F=0.8, CR=0.9).
# Exploration mechanisms: The differential mutation provides global exploration; the
#   random selection of base vectors maintains diversity.
# Exploitation mechanisms: The greedy selection preserves the best-found solutions; the
#   population as a whole converges over generations.
# Boundary handling: Candidate solutions are clipped to the search bounds after mutation
#   and crossover.
# Budget strategy: The initial population is evaluated once, then each generation
#   performs one loop over the entire population, evaluating one trial per target.
#   The loop stops as soon as the evaluation budget is reached.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None.
# Failure modes: Premature convergence due to loss of population diversity; fixed
#   parameters may be suboptimal for certain landscapes; limited budget may prevent
#   reaching the global optimum.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        else:
            # Assume func.bounds.lb / func.bounds.ub
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)

        dim = self.dim
        budget = self.budget

        # Parameters
        F = 0.8           # mutation factor
        CR = 0.9          # crossover rate

        # Population size (at least 4, at most budget//2)
        NP = max(4, min(budget // 2, 10 * dim))

        # Initialize population uniformly in [lb, ub]
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.full(NP, np.inf)

        # Evaluate initial population
        evals = 0
        for i in range(NP):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1

        # Track best
        best_idx = np.argmin(fitness[:evals])
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        while evals < budget:
            # Shuffle target indices each generation
            for i in np.random.permutation(NP):
                if evals >= budget:
                    break

                # Select three distinct random indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                # Mutation
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Crossover (binomial)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])

                # Boundary handling: clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_f = func(trial)
                evals += 1

                # Selection (greedy for minimization)
                if trial_f <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_y:
                        best_y = trial_f
                        best_x = trial.copy()

        return best_x, best_y
