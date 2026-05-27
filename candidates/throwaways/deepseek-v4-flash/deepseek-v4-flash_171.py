import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a classic Differential Evolution (DE/rand/1/bin) with dither for the GNBG benchmark.
#          The algorithm adapts population size to budget and dimension to remain robust across different problem sizes.
# Search state: Maintains a population of candidate vectors and their fitness values. Tracks the best solution found so far.
# Candidate generation: For each target vector, three distinct random population members are selected.
#          A mutant vector is computed as base + F * (second - third). Then binomial crossover with the target produces a trial vector.
# Selection and replacement: Greedy selection: if the trial vector has lower (better) objective value than the target, it replaces the target.
#          The overall best solution is updated after each generation.
# Adaptation: The scale factor F is randomly drawn each generation from [0.5, 0.8] (dither). Crossover rate CR is fixed at 0.9.
#          Population size is set to min(50, max(5, budget//(2*dim))), ensuring at least 4 individuals.
# Exploration mechanisms: Mutation with dither and stochastic crossover encourage exploration. Random initialisation covers the search space.
# Exploitation mechanisms: Greedy selection replaces inferior individuals, and the best solution is stored.
#          Crossover with the target retains good components from the parent.
# Boundary handling: Trial vectors are clipped to the lower/upper bounds.
# Budget strategy: Evaluations are counted precisely. The initial population takes part of the budget; then the loop continues until the budget is exhausted.
# Closest known influences: Standard Differential Evolution (DE/rand/1/bin) with dither, as described by Price, Storn, and Lampinen.
# Novelty or unusual aspects: None – intentionally straightforward and robust.
# Failure modes: May converge prematurely on multimodal functions if population is too small.
#          With very limited budget the algorithm may not make significant progress.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Differential Evolution minimizer for black‑box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Determine population size based on budget and dimension.
        # Keep it small enough to allow several generations, but not too large.
        NP = max(4, min(50, budget // (2 * dim)))
        # Ensure we never allocate more individuals than the budget allows
        if NP > budget:
            NP = budget
        self.NP = NP
        # Fixed parameters
        self.F_low = 0.5
        self.F_high = 0.8
        self.CR = 0.9

    def __call__(self, func):
        # Read bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.array(func.lower, dtype=float)
            upper = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lower = np.array(b.lb, dtype=float)
            upper = np.array(b.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via .lower/.upper or .bounds.lb/.ub")

        dim = self.dim
        NP = self.NP
        # Initialise population uniformly in the domain
        pop = lower + np.random.uniform(size=(NP, dim)) * (upper - lower)

        # Evaluate initial population
        fitness = np.array([func(x) for x in pop])
        evals = NP

        # Track best solution
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main evolutionary loop
        while evals < self.budget:
            # Dither: use a random F per generation
            F = self.F_low + np.random.rand() * (self.F_high - self.F_low)

            # Process each target individual
            for i in range(NP):
                if evals >= self.budget:
                    break

                # Choose three distinct random indices different from i
                candidates = [idx for idx in range(NP) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # Mutation: DE/rand/1
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # Binomial crossover
                cross_points = np.random.rand(dim) < self.CR
                # Ensure at least one component comes from the mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Boundary handling: clip to feasible region
                trial = np.clip(trial, lower, upper)

                # Evaluate trial point
                trial_fitness = func(trial)
                evals += 1

                # Greedy selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
