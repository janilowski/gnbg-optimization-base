import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact Differential Evolution (DE) optimizer for the GNBG
#          black-box minimization benchmark. It uses a steady-state (asynchronous) update scheme
#           with random dithering of the mutation factor.
# Search state: The state is a population of `popsize` candidate solutions (each a `dim`-vector)
#               stored in an array, along with their corresponding objective values.
# Candidate generation: For each parent, a trial vector is created using the rand/1 mutation
#                       (three distinct random population members) followed by binomial crossover
#                       with the parent.
# Selection and replacement: If the trial vector's objective is less than or equal to the parent's,
#                            it replaces the parent immediately (greedy selection).
# Adaptation: The mutation factor F is randomly chosen in [0.5,1.0) at each generation (dithering);
#             crossover rate Cr is fixed at 0.9.
# Exploration mechanisms: The DE mutation difference vector and high Cr promote exploration;
#                         random parent selection gives diversity.
# Exploitation mechanisms: As the population converges, difference vectors shrink, focusing
#                          the search near current best solutions; greedy replacement retains
#                          improving points.
# Boundary handling: Trial vectors are clipped component-wise to the box constraints.
# Budget strategy: The population size is set to `max(10, min(budget//2, 4*dim))` to ensure at
#                  least one full generation after initialization. Candidate evaluations are
#                  performed one by one until the budget is exhausted; partial generations are
#                  allowed.
# Closest known influences: Classic DE/rand/1/bin with dithering (Price et al., 2005).
# Novelty or unusual aspects: None; the implementation follows standard DE principles.
# Failure modes: On very low budgets (e.g., < 20 evaluations) the population may not converge;
#                if the problem is highly multimodal and high-dimensional, DE can get stuck.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Differential Evolution black-box minimizer."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Retrieve bounds (support both attribute styles)
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = getattr(func.bounds, 'lb', getattr(func.bounds, 'lower', None))
            ub = getattr(func.bounds, 'ub', getattr(func.bounds, 'upper', None))
            if lb is None or ub is None:
                raise AttributeError("Cannot determine bounds from function object.")
            lower = np.asarray(lb, dtype=float)
            upper = np.asarray(ub, dtype=float)
        else:
            raise AttributeError("Function object has no lower/upper or bounds attribute.")

        # Ensure bounds are 1D arrays of correct dimension
        if lower.ndim == 0:
            lower = np.full(self.dim, lower)
            upper = np.full(self.dim, upper)

        dim = self.dim
        budget = self.budget

        # Population size: at least 10, at most budget//2, scaled by dim (4*dim heuristic)
        popsize = max(10, min(budget // 2, 4 * dim))

        # Initialize population uniformly within bounds
        pop = np.random.uniform(lower, upper, size=(popsize, dim))
        # Evaluate initial population
        fitness = np.array([func(x) for x in pop])
        evals = popsize  # initial evaluations

        # Track the best solution so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # DE parameters
        Cr = 0.9            # crossover rate

        # Main loop: generate trial vectors one by one until budget exhausted
        while evals < budget:
            # Dithering: choose a random F in [0.5, 1) for this generation
            F = 0.5 + 0.5 * np.random.rand()
            # Process each population member (steady-state)
            for i in range(popsize):
                if evals >= budget:
                    break
                # Choose three distinct random indices different from i
                indices = [j for j in range(popsize) if j != i]
                r1, r2, r3 = np.random.choice(indices, size=3, replace=False)
                # Mutation: trial_base + F * (trial_base - trial_base)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # Binomial crossover with parent pop[i]
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < Cr or j == j_rand:
                        trial[j] = mutant[j]
                # Clipping to bounds
                trial = np.clip(trial, lower, upper)
                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1
                # Greedy selection
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    # Update global best
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
