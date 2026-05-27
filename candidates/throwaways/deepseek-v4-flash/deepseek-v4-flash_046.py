import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) optimizer with adaptive population sizing,
#   designed for black-box minimization. It combines random initialization, mutation,
#   binomial crossover, and greedy selection, and falls back to pure random search when
#   the evaluation budget is too small for a meaningful population.
# Search state: A population of candidate solutions (real-valued vectors) and their objective
#   values. The best solution found so far is tracked separately.
# Candidate generation: For each target vector, a mutant is created by adding the scaled
#   difference between two distinct random population members to a third (DE/rand/1).
#   A trial vector is formed by binomial crossover between the mutant and the target.
# Selection and replacement: Greedy selection: the trial replaces the target if its
#   objective value is strictly lower (minimization). No aging or explicit diversity
#   mechanisms.
# Adaptation: Fixed scaling factor (F=0.8) and crossover rate (CR=0.9). The population
#   size is set once at start based on dimension and budget, and does not adapt during
#   the run.
# Exploration mechanisms: The differential mutation operator and high crossover rate
#   promote exploration, especially in early generations. The population covers the
#   search space via uniform random initialization.
# Exploitation mechanisms: Greedy selection and the continuous refinement of the population
#   through mutation/crossover drive exploitation. Over generations, the population converges.
# Boundary handling: Trial vectors are clipped component-wise to the bounds [lower, upper].
# Budget strategy: All function evaluations (initialisation + generations) are counted.
#   The algorithm stops immediately when the budget is exhausted. For very small budgets,
#   it performs pure random sampling to use every evaluation.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997) with
#   DE/rand/1/bin strategy.
# Novelty or unusual aspects: None; this is a straightforward textbook DE implementation.
#   The fallback to random search for extremely small budgets ensures robustness.
# Failure modes: On highly multimodal or deceptive landscapes, DE/rand/1/bin may converge
#   prematurely due to loss of population diversity. Fixed parameters may not be optimal
#   for all functions. Clipping can cause accumulation at boundaries if the optimum lies
#   inside.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Differential Evolution optimizer for black-box minimization."""

    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # --- Get bounds -------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective function has no lower/upper or bounds attribute")

        # Ensure shape for broadcasting
        lower = lower.reshape(1, -1)
        upper = upper.reshape(1, -1)
        dim = self.dim

        # --- Handle tiny budgets ---------------------------------------------
        # For very small budgets, fall back to pure random search.
        # We need at least 3 individuals for DE/rand/1 and one generation after init.
        min_pop = 3
        if self.budget < min_pop * 2:
            # Random search: evaluate as many points as budget allows
            best_x = None
            best_y = np.inf
            for _ in range(self.budget):
                x = np.random.uniform(lower, upper, size=dim)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x
            return best_x, best_y

        # --- DE population sizing --------------------------------------------
        # Typical: between 4 and 30, proportional to dimension but capped by budget.
        popsize = min(30, max(5, 2 * dim))
        # Ensure we have enough budget for initialisation and at least one generation
        popsize = min(popsize, self.budget // 2)
        if popsize < 3:
            popsize = 3  # but already ensured above

        # --- DE parameters ---------------------------------------------------
        F = 0.8      # scaling factor
        CR = 0.9     # crossover rate

        # --- Initialisation --------------------------------------------------
        population = np.random.uniform(lower, upper, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        evaluations = 0

        # Evaluate initial population
        for i in range(popsize):
            if evaluations >= self.budget:
                break
            fitness[i] = func(population[i])
            evaluations += 1

        # Track best so far
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]

        # --- Main DE loop (DE/rand/1/bin) ------------------------------------
        while evaluations < self.budget:
            for i in range(popsize):
                if evaluations >= self.budget:
                    break

                # Mutation: choose three distinct indices different from i
                candidates = list(range(popsize))
                candidates.remove(i)
                if len(candidates) < 3:
                    break  # should not happen if popsize >= 4
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])

                # Crossover: binomial
                trial = population[i].copy()
                j_rand = np.random.randint(0, dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary handling: clip to bounds
                trial = np.clip(trial, lower, upper)

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
