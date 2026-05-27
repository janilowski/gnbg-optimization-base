# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) optimizer for black-box minimization.
#   Uses DE/rand/1/bin with fixed control parameters and explicit budget management.
# Search state: A population of candidate solutions (vectors), each with its fitness value.
#   The entire population is stored in a 2D numpy array; fitnesses in a 1D array.
#   The best solution found so far (global best) is maintained separately.
# Candidate generation: For each parent index i, three distinct random indices (r0, r1, r2)
#   are chosen from the population, all different from i. A mutant vector v is computed as
#   v = pop[r0] + F * (pop[r1] - pop[r2]), where F = 0.8 is the fixed scaling factor.
#   Then, binomial crossover combines v with the parent: each coordinate is taken from v with
#   probability CR = 0.9, but at least one coordinate is always taken from v (ensured by
#   replacing a random coordinate with the mutant's value).
# Selection and replacement: Greedy selection: if the trial vector has lower (better) objective
#   value than the parent, it replaces the parent in the population. Otherwise the parent is kept.
# Adaptation: No parameter adaptation; F and CR are fixed constants chosen for general robustness.
# Exploration mechanisms: The mutation operator (difference of two random vectors) provides
#   diversity and exploration, especially early in the run. Crossover introduces further mixing.
# Exploitation mechanisms: As the population converges, the differences between individuals shrink,
#   reducing step sizes and focusing search near the current best. The greedy replacement ensures
#   that better solutions persist.
# Boundary handling: After generating the mutant vector v, each component is clipped to the
#   search domain [lower, upper]. The final trial vector is also guaranteed to be within bounds
#   because v was clipped before crossover, and parent is already in bounds.
# Budget strategy: The population size NP is set as max(5, min(10*dim, budget//2)) to fit within
#   the evaluation budget. The number of full generations is computed as (budget - NP) // NP.
#   If NP exceeds budget, NP is reduced to budget (then only initialization occurs). Evaluations
#   are counted precisely; the algorithm stops when the budget would be exceeded.
# Closest known influences: Classic DE/rand/1/bin (Storn & Price, 1997) with fixed parameters.
#   Boundary handling by clipping is a common simple approach.
# Novelty or unusual aspects: None; this implementation follows the standard DE recipe closely.
#   It is intentionally simple, robust, and dimension-agnostic.
# Failure modes: For very low budgets (less than 5 evaluations), only initialization is run.
#   For very high-dimensional functions with small budget, the fixed population size may be too
#   small to maintain diversity, leading to premature convergence. The algorithm may also struggle
#   on extremely multimodal or deceptive landscapes due to fixed control parameters.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """Initialize the DE optimizer with a given evaluation budget and dimension."""
        self.budget = budget
        self.dim = dim

        # Choose population size (NP) to fit within budget while remaining reasonable.
        # Rule: NP = max(5, min(10*dim, budget//2)).
        self.NP = max(5, min(10 * dim, budget // 2))
        # Ensure NP does not exceed budget (otherwise only initialization is possible)
        if self.NP > budget:
            self.NP = budget

        # Fixed DE control parameters (commonly used defaults)
        self.F = 0.8   # mutation scaling factor
        self.CR = 0.9  # crossover probability

        # Compute the number of full generations that can be performed:
        # initial evaluation uses NP evaluations, each generation uses NP evaluations.
        remaining = budget - self.NP
        if remaining < 0:
            self.generations = 0
        else:
            self.generations = remaining // self.NP

    def __call__(self, func):
        """Run the DE optimizer on the given black-box function.

        Args:
            func: A callable that returns a scalar objective value. Must provide lower and
                  upper bounds either as attributes `.lower` / `.upper`, or as `.bounds.lb` / `.bounds.ub`.

        Returns:
            (best_x, best_y): The best solution found and its objective value.
        """
        # --- Read bounds (both common interfaces) ---
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # Assume func.bounds has .lb and .ub attributes
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide .lower/.upper or .bounds.lb/.bounds.ub")

        # Ensure bounds are 1D arrays of length dim
        if lower.ndim == 0:
            lower = np.full(self.dim, lower)
            upper = np.full(self.dim, upper)
        lower = lower.astype(float).ravel()
        upper = upper.astype(float).ravel()

        # --- Initialization ---
        NP = self.NP
        dim = self.dim
        pop = np.random.uniform(lower, upper, size=(NP, dim))
        # Evaluate initial population
        fitness = np.empty(NP)
        evals = 0
        for i in range(NP):
            fitness[i] = func(pop[i])
            evals += 1

        # Track the global best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # --- Main DE loop ---
        # For each generation, we perform NP evaluations (trial vectors)
        for gen in range(self.generations):
            # Loop over each individual (parent)
            for i in range(NP):
                # Select three distinct random indices different from i
                candidates = [j for j in range(NP) if j != i]
                r0, r1, r2 = np.random.choice(candidates, size=3, replace=False)

                # Mutation: DE/rand/1
                mutant = pop[r0] + self.F * (pop[r1] - pop[r2])

                # Boundary handling: clip to domain
                mutant = np.clip(mutant, lower, upper)

                # Crossover: binomial (each dimension with probability CR)
                trial = pop[i].copy()
                cross_points = np.random.rand(dim) < self.CR
                # Ensure at least one dimension comes from the mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                trial[cross_points] = mutant[cross_points]

                # Evaluate trial vector
                trial_fitness = func(trial)
                evals += 1

                # Selection: greedy
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    # Update global best if necessary
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        # --- Final return ---
        return best_x, best_y
