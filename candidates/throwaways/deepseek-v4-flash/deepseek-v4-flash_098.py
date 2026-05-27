import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This implementation uses Differential Evolution (DE/rand/1/bin) to minimize a black-box function (GNBG benchmark). It is compact and relies on standard numpy operations.
# Search state: A population of candidate solutions (real-valued vectors) stored as a 2D array 'pop' of shape (popsize, dim). Their fitness values are stored in 'fitness' array.
# Candidate generation: For each target individual, a mutant vector is created by adding the scaled difference of two other random population members to a third (DE/rand/1). Then binomial crossover combines the mutant with the target to produce a trial vector.
# Selection and replacement: Greedy selection: if the trial vector yields a better (lower) objective value than the target, it replaces the target in the population.
# Adaptation: No parameter adaptation; F (scale factor) and CR (crossover rate) are fixed (0.8 and 0.9, respectively).
# Exploration mechanisms: The mutation operator with random indices and difference vector provides exploration. Crossover also introduces exploration by mixing components.
# Exploitation mechanisms: Selection pressure (better solutions replace worse) and the use of the best-so-far solution stored separately for output. The mutation step size is influenced by the population spread.
# Boundary handling: Trial vectors that exceed the specified lower/upper bounds are clipped to the bounds.
# Budget strategy: The population is initialized using budget // popsize points (if budget limited). The algorithm runs full generations while the remaining budget is at least popsize evaluations; leftover evaluations are not used (budget is not exceeded). The best solution found is returned.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997) with rand/1/bin strategy.
# Novelty or unusual aspects: None – a straightforward DE implementation tailored for black-box optimization with minimal overhead.
# Failure modes: May perform poorly on highly multimodal or deceptive landscapes due to lack of adaptation; fixed mutation and crossover parameters may not suit all problems; population may converge prematurely if spread collapses.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from the function object
        try:
            lb = func.lower
            ub = func.upper
        except AttributeError:
            # Fallback for bbob/COCO style bounds
            lb = func.bounds.lb
            ub = func.bounds.ub
        lb = np.array(lb, dtype=float).ravel()
        ub = np.array(ub, dtype=float).ravel()

        # Population size: scale with dimension but respect budget
        popsize = max(4, min(100, 10 * self.dim))
        popsize = min(self.budget, popsize)

        # Initialize population uniformly in the domain
        pop = np.random.uniform(lb, ub, size=(popsize, self.dim))
        fitness = np.empty(popsize)
        best_x = None
        best_y = np.inf

        # Evaluate initial population
        for i in range(popsize):
            fitness[i] = func(pop[i])
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()

        evals_used = popsize
        remaining = self.budget - evals_used

        # DE parameters
        F = 0.8
        CR = 0.9

        # Main DE loop – run full generations while enough budget remains
        while remaining >= popsize:
            # For each target individual
            for i in range(popsize):
                # Choose three distinct random indices different from i
                candidates = list(range(popsize))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Mutation: DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                trial = pop[i].copy()
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    # Always change at least one component
                    j = np.random.randint(0, self.dim)
                    cross_points[j] = True
                trial[cross_points] = mutant[cross_points]

                # Boundary handling: clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_fitness = func(trial)
                evals_used += 1
                remaining -= 1

                # Greedy selection
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

                # Stop early if budget exhausted within the generation
                if remaining == 0:
                    break

            if remaining == 0:
                break

        # Return the best solution found
        return best_x, best_y
