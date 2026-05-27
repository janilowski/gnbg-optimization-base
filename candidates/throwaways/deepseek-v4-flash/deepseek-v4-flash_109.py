import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A differential evolution (DE) optimizer using rand/1/bin strategy with greedy selection, boundary clipping, and population size adapted to budget and dimension.
# Search state: A population of candidate solutions (real vectors) maintained in an array; also global best.
# Candidate generation: For each target individual, a mutant is created by adding scaled difference of two random distinct population members to a third, then binomial crossover with target to form trial.
# Selection and replacement: Simple greedy replacement: trial replaces target if it yields lower (better) objective value.
# Adaptation: Fixed parameters F=0.5, CR=0.9; population size is heuristically set based on budget and dimension, aiming for at least a few generations.
# Exploration mechanisms: Mutation with difference vectors provides diversity; crossover mixes components.
# Exploitation mechanisms: Selection pressure via greedy replacement and retention of best solution.
# Boundary handling: All candidate vectors are clipped to the provided lower and upper bounds.
# Budget strategy: Precisely tracks evaluation count; stops when budget exhausted; uses random search if budget is too small for a population.
# Closest known influences: Classic DE/rand/1/bin (Storn & Price, 1997).
# Novelty or unusual aspects: Very simple, no adaptive parameter control; population size heuristic to handle limited budgets.
# Failure modes: May converge prematurely or struggle on highly multimodal or deceptive landscapes due to fixed parameters; lacks any restart mechanism; clipping can cause stagnation on boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # Heuristic population size
        # Aim for at least a few generations with reasonable population
        min_popsize = 4
        target_generations = 5
        if budget < 10 * dim:
            # Very low budget: use a small population
            self.popsize = max(min_popsize, min(20, int(budget / 3)))
        else:
            self.popsize = max(min_popsize,
                               min(100, 5 * dim, int(budget / target_generations)))
        # Ensure at least two generations can run
        if self.popsize > budget // 2:
            self.popsize = max(min_popsize, budget // 2)

        # DE parameters (fixed)
        self.CR = 0.9        # Crossover probability
        self.F = 0.5         # Scaling factor

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        else:
            raise AttributeError(
                "Function must provide bounds via lower/upper or bounds.lb/ub")
        dim = self.dim
        budget = self.budget
        popsize = self.popsize

        # Fallback to pure random search when budget too small
        if budget <= popsize:
            best_x = None
            best_y = np.inf
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x
            return (best_x, best_y)

        # Initial population
        population = np.random.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        for i in range(popsize):
            fitness[i] = func(population[i])
        evals = popsize

        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        while evals < budget:
            remaining = budget - evals
            gen_trials = min(popsize, remaining)   # trials we can still afford
            if gen_trials <= 0:
                break

            for i in range(gen_trials):
                # Choose three distinct indices different from i
                indices = list(range(popsize))
                indices.remove(i)
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)

                # Mutation: DE/rand/1
                mutant = population[r1] + self.F * (population[r2] - population[r3])
                # Boundary clipping
                mutant = np.clip(mutant, lb, ub)

                # Binomial crossover
                cross_points = np.random.rand(dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluation and selection
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return (best_x, best_y)
