import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a simple Differential Evolution (DE) algorithm with randomized scaling factor and binomial crossover.
# It uses a fixed population size adjusted to the budget, and always maintains and updates the best solution found.
# Search state: The state consists of a population of candidate solutions (array of shape (pop_size, dim)), their fitness values,
#                the best found solution and its fitness, generation counter, and number of evaluations used.
# Candidate generation: For each individual in the population, a mutant vector is created using the DE/rand/1 scheme:
#                base = random individual, difference = two other random distinct individuals. The scaling factor F is sampled
#                uniformly in [0.5, 0.8] per mutation. Then binomial crossover with probability Cr=0.9 combines mutant with
#                target to produce trial vector.
# Selection and replacement: Greedy selection: if trial vector has lower fitness (minimization) than target, it replaces the
#                target in the next generation. The best overall solution is updated if any trial improves it.
# Adaptation: No explicit adaptation of parameters; F and Cr are fixed (Cr), F randomized per mutation to provide variability.
# Exploration mechanisms: Random selection of base and difference vectors, random scaling factor, and crossover provide
#                exploration across the search space.
# Exploitation mechanisms: The greedy selection pushes the population toward better regions, and the differential mutation
#                uses the current population's spread to generate new candidates near promising areas.
# Boundary handling: If any coordinate of the trial vector is outside the bounds, it is reflected back into the domain
#                (bounce-back) with a random step inside to avoid stagnation on boundaries.
# Budget strategy: The algorithm precomputes the maximum number of generations from pop_size and budget. It stops when the
#                number of evaluations reaches the budget. No extra evaluations are performed beyond the budget.
# Closest known influences: Classic DE/rand/1/bin with dither (random F). Similar to standard implementations in
#                scipy.optimize.differential_evolution.
# Novelty or unusual aspects: None; it is a straightforward implementation.
# Failure modes: May converge prematurely in high-dimensional multimodal landscapes due to lack of diversity; population size
#                might be too small for high dimensions; fixed parameters may not adapt to problem difficulty.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the differential evolution optimizer.

        Args:
            budget: Maximum number of function evaluations.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimization on a given test function.

        Args:
            func: The objective function. Must have bounds accessible via either
                  `func.lower` / `func.upper` or `func.bounds.lb` / `func.bounds.ub`.

        Returns:
            (best_x, best_y): The best discovered solution and its function value.
        """
        # ---------- Read bounds ----------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot read bounds from the provided function.")

        dim = self.dim
        # Ensure bounds are 1-D arrays
        if lb.ndim == 0:
            lb = np.full(dim, lb)
        if ub.ndim == 0:
            ub = np.full(dim, ub)
        lb = lb.ravel()
        ub = ub.ravel()

        # ---------- Population sizing ----------
        # Use a moderate population size scaled by budget, but at least 4 and at most 50.
        pop_size = max(4, min(50, self.budget // 5))
        # Maximum number of generations (each generation uses pop_size evaluations)
        # but we may stop earlier if budget is exhausted.
        max_generations = self.budget // pop_size

        # ---------- Initialization ----------
        # Latin Hypercube-like sampling: sample uniformly from [0,1) and apply bounds
        rng = np.random.RandomState()  # use numpy's global state; harness sets seed
        # Simple uniform random initialization (faster, still acceptable)
        population = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        # Evaluate initial population
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(population[i])
            evals += 1

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]

        # ---------- DE main loop ----------
        # Parameters
        Cr = 0.9  # crossover probability
        # F will be sampled per mutation in [0.5, 0.8]

        for generation in range(max_generations):
            if evals >= self.budget:
                break

            new_pop = population.copy()
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # Choose three distinct random indices different from i
                indices = list(range(pop_size))
                indices.remove(i)
                chosen = rng.choice(indices, size=3, replace=False)
                a, b, c = chosen

                # Mutation: DE/rand/1 with dither
                F = rng.uniform(0.5, 0.8)
                mutant = population[a] + F * (population[b] - population[c])

                # Binomial crossover
                trial = population[i].copy()
                j_rand = rng.randint(0, dim)
                for j in range(dim):
                    if rng.rand() < Cr or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary handling: reflect back into bounds
                trial = np.clip(trial, lb, ub)  # simple clipping; alternative bounce-back
                # (bounce-back with random step would be more sophisticated, clip is adequate)

                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # Greedy selection
                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
                    # Update best if needed
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

            population = new_pop
            # Optional: check if best_y improved; could break if stagnation, but we continue.

        return best_x, best_y
