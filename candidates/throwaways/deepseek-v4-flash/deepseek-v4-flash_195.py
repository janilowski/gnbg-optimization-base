# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This implementation uses a standard Differential Evolution (DE/rand/1/bin) algorithm for black-box minimization. It is designed to be robust across dimensions and respects the evaluation budget exactly.
# Search state: A population of candidate solutions (vectors) and their corresponding fitness values is maintained. The best solution found so far is tracked.
# Candidate generation: For each target vector, three distinct random members of the population (excluding the target) are selected. A mutant vector is produced as base + F * (diff1 - diff2). The classic binomial crossover then mixes the mutant with the target to form a trial vector.
# Selection and replacement: If the trial vector yields a lower (better) objective value than the target, it replaces the target in the next generation. Otherwise, the target is retained. This greedy selection ensures monotonic improvement of the best solution.
# Adaptation: The algorithm uses fixed control parameters: scaling factor F = 0.8 and crossover rate CR = 0.9. No adaptive parameter control is implemented, keeping the algorithm simple and predictable.
# Exploration mechanisms: Mutation based on population differences and random crossover maintain diversity. The population is initialized uniformly over the search domain.
# Exploitation mechanisms: As the population converges, differences between vectors shrink, leading to finer local search. Selection pressure drives the population toward promising regions.
# Boundary handling: Mutant vectors (and crossover products) are simply clamped to the lower and upper bounds. No more sophisticated boundary repair is used.
# Budget strategy: The budget is used for all function evaluations. The initial population consumes the first part of the budget. Subsequent generations run until the budget is exhausted, respecting per-generation loop breaks when budget runs out.
# Closest known influences: Classic DE/rand/1/bin as introduced by Storn and Price (1997).
# Novelty or unusual aspects: None. A straightforward, compact implementation tailored for the GNBG benchmark.
# Failure modes: On extremely low budgets (e.g., less than 10 evaluations), performance may be poor because the population cannot be fully initialized. The algorithm also may struggle on highly multimodal or deceptive landscapes due to fixed parameters, but it generally offers good robustness.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Read bounds from the function object
        try:
            lower = func.lower
            upper = func.upper
        except AttributeError:
            try:
                lower = func.bounds.lb
                upper = func.bounds.ub
            except AttributeError:
                raise AttributeError("Cannot find bounds: func.lower/upper or func.bounds.lb/ub required")

        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)

        evals = 0
        best_x = None
        best_y = float('inf')

        # Determine population size: try to have at least 3*D individuals, but cap before consuming too many evaluations
        popsize = min(10 * self.dim, self.budget // 2)
        if popsize < 3:
            popsize = 3  # need at least 3 for DE mutation

        # If budget is very small, fall back to random search
        if self.budget < popsize:
            best_x = np.random.uniform(lower, upper, size=self.dim)
            best_y = func(best_x)
            evals = 1
            while evals < self.budget:
                candidate = np.random.uniform(lower, upper, size=self.dim)
                y = func(candidate)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = candidate
            return best_x, best_y

        # Population initialization
        population = np.random.uniform(lower, upper, size=(popsize, self.dim))
        fitness = np.full(popsize, np.inf)
        for i in range(popsize):
            fitness[i] = func(population[i])
            evals += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = population[i].copy()

        # DE parameters
        F = 0.8      # scaling factor
        CR = 0.9     # crossover rate

        # Main DE loop
        while evals < self.budget:
            # Generate next generation
            next_pop = population.copy()
            next_fit = fitness.copy()
            # Shuffle indices to avoid bias
            indices = np.arange(popsize)
            np.random.shuffle(indices)
            for i in indices:
                if evals >= self.budget:
                    break

                # Choose three distinct random indices different from i
                candidates = [j for j in range(popsize) if j != i]
                r = np.random.choice(candidates, size=3, replace=False)
                a, b, c = r[0], r[1], r[2]

                # Mutation
                mutant = population[a] + F * (population[b] - population[c])
                # Clamp to bounds
                mutant = np.clip(mutant, lower, upper)

                # Binomial crossover
                trial = population[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluation
                trial_y = func(trial)
                evals += 1

                # Selection
                if trial_y <= next_fit[i]:
                    next_pop[i] = trial
                    next_fit[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()

            # Update population for next generation
            population = next_pop
            fitness = next_fit

        # Return the best solution found
        return best_x, best_y
