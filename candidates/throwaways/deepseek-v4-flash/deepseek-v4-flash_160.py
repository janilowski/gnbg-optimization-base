import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements Differential Evolution (DE) with rand/1/bin strategy for black-box minimization.
# Search state: A population of candidate solutions uniformly sampled from the search space.
# Candidate generation: For each target vector, a mutant is created by adding a scaled difference of two other random population vectors (rand/1). Then binomial crossover combines the mutant with the target.
# Selection and replacement: Greedy selection – the trial replaces the target if and only if its objective value is lower (minimization).
# Adaptation: No adaptive parameters; fixed mutation factor F=0.8 and crossover probability CR=0.9.
# Exploration mechanisms: Mutation explores by adding random differences, crossover mixes dimensions; population diversity maintains exploration.
# Exploitation mechanisms: Selection favors better solutions; best-so-far solution is tracked separately.
# Boundary handling: Mutated vectors are clipped to the variable bounds.
# Budget strategy: Allocates evaluations to initial population and then iterates generation by generation until budget exhausted; if budget is insufficient for a full population, falls back to random search.
# Closest known influences: Standard differential evolution (Price, Storn, Lampinen), rand/1/bin variant.
# Novelty or unusual aspects: Simple and straightforward implementation with no external dependencies beyond numpy.
# Failure modes: May converge prematurely on highly multimodal functions; fixed parameters may not be optimal for all landscapes; boundary clipping can distort mutation steps near edges.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Read bounds from func
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.array(func.lower, dtype=float)
            upper = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
            lower, upper = lb, ub
        else:
            raise AttributeError("Cannot read bounds from func")

        dim = self.dim
        budget = self.budget

        # Fallback for very small budgets: pure random search
        if budget < 5:
            best_x = None
            best_y = np.inf
            for _ in range(budget):
                x = lower + (upper - lower) * np.random.rand(dim)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x
            return best_x, best_y

        # Population size: at least 10, at most budget//2, roughly 5*dim capped
        popsize = min(budget // 2, max(10, int(5 * dim)))
        if popsize < 4:
            popsize = 4

        # Initialise population uniformly
        pop = lower + (upper - lower) * np.random.rand(popsize, dim)
        pop_fitness = np.array([func(pop[i]) for i in range(popsize)])
        fevals = popsize

        # Track best so far
        best_idx = np.argmin(pop_fitness)
        best_x = pop[best_idx].copy()
        best_y = pop_fitness[best_idx]

        # DE parameters
        F = 0.8
        CR = 0.9

        while fevals < budget:
            for i in range(popsize):
                if fevals >= budget:
                    break

                # Select three distinct random indices != i
                idx_pool = [j for j in range(popsize) if j != i]
                a, b, c = np.random.choice(idx_pool, size=3, replace=False)

                # Mutation: rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Boundary clipping
                mutant = np.clip(mutant, lower, upper)

                # Binomial crossover
                jrand = np.random.randint(dim)
                crossover_mask = np.random.rand(dim) < CR
                crossover_mask[jrand] = True
                trial = np.where(crossover_mask, mutant, pop[i])

                # Evaluate trial
                trial_fitness = func(trial)
                fevals += 1

                # Greedy selection
                if trial_fitness < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
