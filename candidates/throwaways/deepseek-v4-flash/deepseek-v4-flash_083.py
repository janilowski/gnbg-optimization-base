import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact implementation of the Differential Evolution (DE/rand/1/bin) algorithm
#          for black-box minimization. It adapts population size to the budget and dimension.
# Search state: A population of candidate solution vectors stored as a 2D numpy array,
#               and their corresponding objective function values stored as a 1D array.
# Candidate generation: For each parent, a mutant vector is created by adding the scaled
#                       difference of two random distinct population members to a third,
#                       then binomial crossover with the parent to produce a trial vector.
# Selection and replacement: Greedy selection: the trial replaces the parent if its fitness
#                            is better (lower).
# Adaptation: No parameter adaptation; fixed mutation factor F and crossover rate CR.
# Exploration mechanisms: Mutation with a fixed scaling factor (F=0.8) introduces diversity;
#                         binomial crossover with high CR (0.9) mixes components from the mutant.
# Exploitation mechanisms: Greedy replacement ensures that only improvements are kept; the
#                          population gradually converges to better regions.
# Boundary handling: Trial vectors are clipped to the search bounds [lower, upper].
# Budget strategy: The population is evaluated once at initialization; each generation performs
#                  one evaluation per individual (population_size evaluations). The process stops
#                  exactly when the cumulative evaluation count equals the budget.
# Closest known influences: Standard Differential Evolution (DE/rand/1/bin).
# Novelty or unusual aspects: None; a straightforward implementation emphasizing readability.
# Failure modes: May stagnate on highly multimodal or high-dimensional problems if the fixed
#                parameters cause premature loss of diversity; poor performance on very low
#                budgets.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the algorithm with an evaluation budget and problem dimension.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimizer on the given objective function.

        Parameters
        ----------
        func : callable
            The objective function, with attributes lower/upper or bounds.lb/bounds.ub.

        Returns
        -------
        best_x : numpy.ndarray (1D)
            The best solution found.
        best_y : float
            The objective value at best_x.
        """
        # --------------------------------------------------------------------
        # 1. Determine search bounds
        # --------------------------------------------------------------------
        try:
            lower = np.asfarray(func.lower, dtype=float)
            upper = np.asfarray(func.upper, dtype=float)
        except AttributeError:
            lower = np.asfarray(func.bounds.lb, dtype=float)
            upper = np.asfarray(func.bounds.ub, dtype=float)

        # Ensure both are 1D arrays
        lower = lower.ravel()
        upper = upper.ravel()
        dim = self.dim

        # --------------------------------------------------------------------
        # 2. Population size and algorithm parameters
        # --------------------------------------------------------------------
        # Heuristic: population size = min(20*dim, budget//2) but at least 4
        pop_size = min(20 * dim, self.budget // 2)
        pop_size = max(pop_size, 4)

        # Fixed DE parameters
        F = 0.8       # mutation scaling factor
        CR = 0.9      # crossover rate

        # --------------------------------------------------------------------
        # 3. Initialise population uniformly in the bounds
        # --------------------------------------------------------------------
        pop = lower + np.random.rand(pop_size, dim) * (upper - lower)
        fitness = np.full(pop_size, np.inf)
        evaluations = 0

        # Evaluate initial population
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evaluations += 1
            if evaluations >= self.budget:
                # Budget exhausted during initialisation – return best so far
                best_idx = np.argmin(fitness)
                return pop[best_idx].copy(), fitness[best_idx]

        # Track the best solution
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # --------------------------------------------------------------------
        # 4. Main DE loop
        # --------------------------------------------------------------------
        while evaluations < self.budget:
            # For each individual, generate a trial vector
            for i in range(pop_size):
                if evaluations >= self.budget:
                    break

                # Mutation: choose three distinct random indices != i
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(indices, size=3, replace=False)

                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary handling: clip to [lower, upper]
                trial = np.clip(trial, lower, upper)

                # Evaluate trial
                trial_fitness = func(trial)
                evaluations += 1

                # Selection: replace if better
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

            # Update best index after full generation (if budget remains)
            if evaluations < self.budget:
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < best_y:
                    best_y = fitness[best_idx]
                    best_x = pop[best_idx].copy()

        return best_x, best_y
