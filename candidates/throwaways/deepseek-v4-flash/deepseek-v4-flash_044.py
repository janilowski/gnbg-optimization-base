import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements Differential Evolution (DE/rand/1/bin) with reflection boundary handling.
# Search state: Population of candidate solutions (size N) and their fitness values.
# Candidate generation: For each target vector, mutant = x_r1 + F * (x_r2 - x_r3) with distinct random indices, then binomial crossover with probability CR to produce trial.
# Selection and replacement: Greedy – trial replaces target if strictly better (lower fitness).
# Adaptation: None (fixed F=0.5, CR=0.9). Population size scales with dimension and budget.
# Exploration mechanisms: Large differential weight F and high crossover rate CR promote diversity.
# Exploitation mechanisms: Greedy selection retains best solutions; mutation step size depends on population spread.
# Boundary handling: Reflected coordinates back into the feasible domain.
# Budget strategy: Population size N ≈ min(40, max(4*dim, budget//2, 4)), ensures at least a few generations possible.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: Simple, no adaptation, suitable for moderate budgets.
# Failure modes: Premature convergence on multimodal landscapes; poor performance when budget is very low.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.array(func.lower).astype(float)
            upper = np.array(func.upper).astype(float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lower = np.array(func.bounds.lb).astype(float)
            upper = np.array(func.bounds.ub).astype(float)
        else:
            raise ValueError("Cannot find bounds from function object")

        dim = self.dim
        # Population size: at least 4, at most 40, scaled with dimension
        N = max(4, min(40, 4 * dim, self.budget // 2))
        # Ensure at least one generation can happen
        if N > self.budget:
            N = self.budget

        # Parameters
        F = 0.5    # differential weight
        CR = 0.9   # crossover probability

        # Initialize population uniformly in bounds
        pop = np.random.uniform(low=lower, high=upper, size=(N, dim))
        # Evaluate population
        fitness = np.array([func(x) for x in pop])
        evals = N
        # Keep track of best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main loop
        generation = 0
        while evals < self.budget:
            generation += 1
            for i in range(N):
                if evals >= self.budget:
                    break

                target = pop[i]
                # Choose three distinct random indices different from i
                indices = list(range(N))
                indices.remove(i)
                chosen = np.random.choice(indices, size=3, replace=False)
                r1, r2, r3 = chosen
                # Mutate
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, target)
                # Ensure at least one dimension from mutant
                trial[j_rand] = mutant[j_rand]
                # Boundary reflection
                trial = np.where(trial < lower, 2 * lower - trial, trial)
                trial = np.where(trial > upper, 2 * upper - trial, trial)
                # Clip to avoid numerical overflow (though reflection should keep inside)
                trial = np.clip(trial, lower, upper)

                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    # Update global best if necessary
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

            # Optional: check if we used all evaluations inside loop
            if evals >= self.budget:
                break

        return best_x, best_y
