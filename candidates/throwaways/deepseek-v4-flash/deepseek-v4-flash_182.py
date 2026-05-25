# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Standard Differential Evolution (DE/rand/1/bin) for black-box minimization.
# Search state: A population of candidate solutions (np.ndarray of shape (popsize, dim)) and the best solution found so far.
# Candidate generation: For each parent vector, a mutant vector = base + F * (diff1 - diff2), then binomial crossover with probability CR to produce trial vector.
# Selection and replacement: One-to-one greedy selection: trial replaces parent if trial objective <= parent objective (minimization).
# Adaptation: No adaptation; fixed parameters (F=0.7, CR=0.9). Population size is set as a function of dimension and budget.
# Exploration mechanisms: Mutation uses scaled random differences between population members, enabling broad search.
# Exploitation mechanisms: Crossover mixes mutant and parent components; greedy selection drives towards better solutions.
# Boundary handling: After mutation and crossover, trial vectors are clipped to the variable bounds.
# Budget strategy: All evaluations are counted precisely. The algorithm stops when the next evaluation would exceed the budget.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None; straightforward implementation.
# Failure modes: In high dimensions with limited budget, population may not converge; fixed parameters may not suit all landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Differential Evolution (DE/rand/1/bin) for minimization with a limited evaluation budget."""

    def __init__(self, budget: int, dim: int):
        """
        Args:
            budget: Maximum number of function evaluations allowed.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """Run the algorithm on the given objective function.

        Args:
            func: Callable with attributes .lower/.upper or .bounds.lb/.bounds.ub
                  to define the box constraints.

        Returns:
            best_x: Best solution found (numpy array of shape (dim,)).
            best_y: Best objective value (float).
        """
        # --- Read bounds ---
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        # Ensure 1-D arrays
        lb = lb.flatten()
        ub = ub.flatten()
        dim = self.dim
        if len(lb) != dim:
            raise ValueError("Lower bound dimension mismatch")
        if len(ub) != dim:
            raise ValueError("Upper bound dimension mismatch")

        # --- Population size ---
        # At least 4, at most budget // 2, and typical DE recommendation of 10*dim.
        popsize = max(4, min(10 * dim, self.budget // 2))

        # --- DE parameters ---
        F = 0.7      # scaling factor
        CR = 0.9     # crossover probability

        # --- Initialization ---
        # Uniform random population within bounds
        population = np.random.uniform(lb, ub, size=(popsize, dim))

        # Evaluate initial population
        evaluations = 0
        fitness = np.empty(popsize)
        for i in range(popsize):
            fitness[i] = func(population[i])
            evaluations += 1
            if evaluations >= self.budget:
                break  # budget exhausted during initialization (unlikely but possible)
        else:
            # all initial evaluations done
            pass

        # Best so far
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]

        # Main loop: generate trial vectors while evaluations < budget
        while evaluations < self.budget:
            # For each individual in the population
            for i in range(popsize):
                if evaluations >= self.budget:
                    break

                # --- Mutation: pick three distinct random indices different from i ---
                candidates = list(range(popsize))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                # base vector (population[a]), difference vectors (b and c)
                mutant = population[a] + F * (population[b] - population[c])

                # --- Crossover: binomial ---
                trial = np.where(
                    np.random.rand(dim) < CR,
                    mutant,
                    population[i]
                )
                # Ensure at least one component from mutant (standard DE trick)
                j_rand = np.random.randint(0, dim)
                trial[j_rand] = mutant[j_rand]

                # --- Boundary handling: clip to bounds ---
                trial = np.clip(trial, lb, ub)

                # --- Evaluation ---
                trial_fitness = func(trial)
                evaluations += 1

                # --- Selection: greedy replacement ---
                if trial_fitness <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
