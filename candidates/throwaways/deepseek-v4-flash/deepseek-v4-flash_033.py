import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple implementation of Differential Evolution (DE/rand/1/bin) for black-box minimization.
# Search state: A population of candidate vectors, each with its current fitness value, plus the overall best solution.
# Candidate generation: For each target vector, a donor is created by adding a scaled difference of two distinct random population vectors to a third distinct vector.
# Selection and replacement: After binomial crossover, trial vectors replace their target if they yield lower (better) fitness.
# Adaptation: Fixed mutation scaling factor F=0.8 and crossover rate CR=0.9; population size is dynamically chosen based on budget and dimension.
# Exploration mechanisms: Random initial population, mutation using random differential vectors, and binomial crossover that can replace many coordinates.
# Exploitation mechanisms: Greedy selection (keep better solution), recording of the best ever found; mutation step size is moderate.
# Boundary handling: Violated coordinates are reflected back into the domain (bounce-back).
# Budget strategy: The algorithm stops exactly when the evaluation counter reaches the budget; no extra evaluations are made.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: Very simple, with no adaptive parameter control; population size is chosen heuristically to fit within the budget.
# Failure modes: Early convergence on highly multimodal problems; low budget may lead to poor exploration; fixed F and CR may be suboptimal for certain problem characteristics.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        try:
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        except AttributeError:
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)

        dim = self.dim
        budget = self.budget

        # Special case: budget too small for a population
        if budget < 4:
            best_x = None
            best_y = np.inf
            for _ in range(budget):
                x = lb + np.random.uniform(0, 1, size=dim) * (ub - lb)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # Determine population size heuristically
        # At least one full generation (pop_size + pop_size*pop_size?) Actually DE evaluation counts:
        # initialization: pop_size evals; each generation: pop_size evals (one per target).
        # We want at least 2 generations (including init). So pop_size <= budget // 2.
        # Also keep it between 4 and 10*dim.
        max_possible = budget // 2
        pop_size = max(4, min(10 * dim, max_possible))
        # Ensure we don't exceed budget with initialization
        if pop_size > budget:
            pop_size = budget

        # DE parameters
        F = 0.8
        CR = 0.9

        # Initialize population uniformly within bounds
        pop = lb + np.random.uniform(0, 1, size=(pop_size, dim)) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = pop[0].copy()
        best_y = np.inf
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            y = func(pop[i])
            fitness[i] = y
            evals += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()
            if evals >= budget:
                # Budget exhausted during initialization (unlikely with pop_size <= budget)
                return best_x, best_y

        # Main DE loop
        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Choose three distinct indices different from i
                indices = [j for j in range(pop_size) if j != i]
                chosen = np.random.choice(indices, 3, replace=False)
                a, b, c = chosen

                # Mutation
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Crossover: binomial, ensure at least one coordinate from mutant
                cross_mask = np.random.rand(dim) < CR
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(0, dim)] = True
                trial = np.where(cross_mask, mutant, pop[i])

                # Boundary handling: reflect
                for d in range(dim):
                    if trial[d] < lb[d]:
                        trial[d] = lb[d] + (lb[d] - trial[d])
                        if trial[d] > ub[d]:
                            trial[d] = lb[d] + np.random.rand() * (ub[d] - lb[d])  # fallback
                    elif trial[d] > ub[d]:
                        trial[d] = ub[d] - (trial[d] - ub[d])
                        if trial[d] < lb[d]:
                            trial[d] = lb[d] + np.random.rand() * (ub[d] - lb[d])
                    # Clamp just in case
                    trial[d] = np.clip(trial[d], lb[d], ub[d])

                # Evaluate trial
                y = func(trial)
                evals += 1
                if y < fitness[i]:
                    pop[i] = trial
                    fitness[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

        return best_x, best_y
