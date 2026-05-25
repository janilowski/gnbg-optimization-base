# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a Differential Evolution (DE) optimizer for black-box minimization.
# Search state: The algorithm maintains a population of candidate solutions (real vectors) in the search space.
# Candidate generation: Offspring (trial vectors) are created using the DE/rand/1/bin scheme: for each target vector, three distinct random population members are selected, a mutant is formed by adding a scaled difference vector, and then binomial crossover with the target produces the trial.
# Selection and replacement: Greedy selection: each trial vector replaces its parent if it yields a lower (better) objective function value.
# Exploration mechanisms: The differential mutation uses a scaling factor F (typically in [0.5, 1.0]) to control the exploration radius; crossover probability CR determines the fraction of dimensions inherited from the mutant, promoting diversity.
# Exploitation mechanisms: As the population converges, difference vectors shrink, focusing the search around promising regions; greedy selection ensures only improvements survive.
# Boundary handling: Trial vectors that violate bounds are repaired by reflecting the component inside the feasible region or clamping to the nearest bound (hybrid approach).
# Budget strategy: The optimizer stops exactly when the total number of objective evaluations reaches the allocated budget; it returns the best solution found so far.
# Closest known influences: Classic DE (Storn & Price, 1997) with the DE/rand/1/bin variant.
# Novelty or unusual aspects: None; this is a straightforward implementation of standard DE adapted for the benchmarking harness.
# Failure modes: Premature convergence on multimodal landscapes if population diversity is lost; may fail to find the global optimum within tight budgets; performance depends on the choice of F and CR (fixed here to typical values). 
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Black-box minimizer using Differential Evolution (DE/rand/1/bin).
    """

    def __init__(self, budget: int, dim: int):
        """
        Initialize the optimizer.

        Args:
            budget: Maximum number of function evaluations.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimization.

        Args:
            func: Objective function with attributes lower/upper (or bounds.lb/bounds.ub).

        Returns:
            (best_x, best_y): Best solution found and its objective value.
        """
        # --- Read bounds ----------------------------------------------------
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            # fallback to func.bounds
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        dim = self.dim
        budget = self.budget

        # --- Configuration --------------------------------------------------
        pop_size = max(10, int(4 * dim))          # population size (rule of thumb)
        F = 0.8                                   # scaling factor
        CR = 0.9                                  # crossover probability

        # --- Initialization ------------------------------------------------
        population = np.random.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_y = np.inf
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            fitness[i] = func(population[i])
            evals += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = population[i].copy()

        # --- Main DE loop --------------------------------------------------
        while evals < budget:
            for i in range(pop_size):
                # Select three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutate
                mutant = population[r1] + F * (population[r2] - population[r3])

                # Binomial crossover
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary handling: reflect or clamp
                trial = np.clip(trial, lb, ub)
                # If a component is still outside, reflect (improved handling)
                below = trial < lb
                above = trial > ub
                trial[below] = 2 * lb[below] - trial[below]
                trial[above] = 2 * ub[above] - trial[above]
                trial = np.clip(trial, lb, ub)  # ensure final safety

                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # Greedy selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

                # Check budget after each evaluation
                if evals >= budget:
                    break

        return best_x, best_y
