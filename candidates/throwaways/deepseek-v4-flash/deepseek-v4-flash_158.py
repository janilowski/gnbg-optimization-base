import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE/rand/1/bin) with reflection boundary handling.
# Search state: A population of candidate solutions (vectors) and their objective values.
# Candidate generation: For each parent, a mutant vector is created by adding the scaled
#   difference of two randomly chosen population members to a third distinct member.
#   The trial vector is formed by binomial crossover between the parent and mutant.
# Selection and replacement: Greedy: the trial replaces the parent if it yields a lower
#   objective value (minimization).
# Adaptation: None; fixed crossover rate CR=0.9 and scaling factor F=0.8.
# Exploration mechanisms: Mutation with random vector differences provides diversity;
#   binomial crossover mixes components to explore the search space.
# Exploitation mechanisms: Greedy selection retains improvements, and the population
#   gradually converges toward better regions.
# Boundary handling: Reflection (mirror) – any component of the trial vector that lies
#   outside [lb, ub] is reflected back into the bounds.
# Budget strategy: The population size is fixed; the algorithm runs generation by
#   generation until the total number of function evaluations reaches the budget.
# Closest known influences: Classic differential evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None; straightforward implementation for benchmark use.
# Failure modes: May converge prematurely on multimodal landscapes; poor performance
#   when the budget is too small to reach a good solution.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from the provided function object
        if hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)

        # Population size: a simple heuristic based on dimension
        pop_size = min(max(4 * self.dim, 10), self.budget // 2)
        if pop_size < 2:
            pop_size = 2

        # DE parameters
        F = 0.8      # scaling factor
        CR = 0.9     # crossover rate

        # Initialize population uniformly in bounds
        pop = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_y = np.inf
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()
            if evals >= self.budget:
                # Budget exhausted during initialization
                return best_x, best_y

        # Main DE loop
        while evals < self.budget:
            for i in range(pop_size):
                # Select three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutation: v = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # Crossover: binomial trial vector
                cross_points = np.random.rand(self.dim) < CR
                # Ensure at least one component is inherited from mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Boundary reflection
                # Reflect components that are outside [lb, ub]
                # For each dimension, if below lb, mirror: lb + (lb - trial)
                # If above ub, mirror: ub - (trial - ub)
                # Use while loop to handle repeated reflections in case of large overstep
                # (simple reflection usually suffices for typical mutation sizes)
                reflection_needed = (trial < lb) | (trial > ub)
                if np.any(reflection_needed):
                    for j in range(self.dim):
                        if trial[j] < lb[j]:
                            trial[j] = lb[j] + (lb[j] - trial[j])
                        elif trial[j] > ub[j]:
                            trial[j] = ub[j] - (trial[j] - ub[j])
                        # Clamp to avoid numerical issues after reflection
                        if trial[j] < lb[j]:
                            trial[j] = lb[j]
                        if trial[j] > ub[j]:
                            trial[j] = ub[j]

                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # Selection (greedy)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

                # Stop if budget exhausted
                if evals >= self.budget:
                    return best_x, best_y

        return best_x, best_y
