# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A differential evolution (DE) optimizer with fixed parameters and a fallback random search for very small budgets. It handles continuous black-box minimization under a limited evaluation budget.
# Search state: A population of candidate solutions and their objective values; the best found solution (x, y) is tracked globally.
# Candidate generation: For each population member, a trial vector is created using DE/rand/1 mutation and binomial crossover. Mutation uses three distinct random individuals drawn from the current population.
# Selection and replacement: Greedy selection: if the trial yields a lower objective value than the current member, it replaces that member.
# Adaptation: No parameter adaptation; F = 0.5 and CR = 0.9 are fixed.
# Exploration mechanisms: The difference vector in mutation and the crossover operation promote global exploration across the search space.
# Exploitation mechanisms: Greedy selection and the gradual replacement of poor individuals drive the population toward better regions; the best ever solution is preserved.
# Boundary handling: Out-of-bounds coordinates are reflected back into the feasible domain (mirror reflection).
# Budget strategy: Budget is split into an initial population evaluation and a fixed number of generations. If the budget is very small (<20 evaluations), the algorithm falls back to uniform random search.
# Closest known influences: Classic DE/rand/1/bin algorithm (Storn & Price, 1997).
# Novelty or unusual aspects: None; this is a straightforward, minimal implementation of DE.
# Failure modes: The fixed parameters may be suboptimal for some problems; the method can stagnate on highly multimodal landscapes; the population size (capped at 30) may be too small for high-dimensional problems; greedy selection may lose diversity early.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # Determine bounds from the function object (supports two common interfaces)
        try:
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        except AttributeError:
            try:
                lb = np.array(func.bounds.lb)
                ub = np.array(func.bounds.ub)
            except AttributeError:
                raise ValueError("Cannot locate lower/upper bounds from the function object.")

        budget = self.budget
        dim = self.dim

        # Fallback: if budget is very small, use pure random search
        if budget < 20:
            best_x = None
            best_y = float('inf')
            for _ in range(budget):
                x = lb + (ub - lb) * np.random.rand(dim)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return (best_x, best_y)

        # Set population size (min 4, max 30, but no more than half the budget)
        pop_size = max(4, min(30, budget // 2))
        # Number of generations we can afford after initial evaluation
        max_generations = (budget - pop_size) // pop_size

        # Initialise population uniformly within bounds
        pop = lb + (ub - lb) * np.random.rand(pop_size, dim)
        fit = np.full(pop_size, float('inf'))

        # Evaluate initial population
        for i in range(pop_size):
            fit[i] = func(pop[i])

        # Track best overall
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # DE parameters (fixed)
        F = 0.5
        CR = 0.9

        for _ in range(max_generations):
            for i in range(pop_size):
                # Choose three distinct indices different from i
                choices = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(choices, size=3, replace=False)

                # Mutation: DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                cross_points = np.random.rand(dim) < CR
                # Ensure at least one dimension is crossed over
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True

                trial = np.where(cross_points, mutant, pop[i])

                # Boundary handling: reflect out-of-bound coordinates back inside
                # Reflection formula: if x < lb, set to 2*lb - x; if x > ub, set to 2*ub - x
                # Then clamp in case reflection still leaves out (e.g., near boundary)
                for d in range(dim):
                    if trial[d] < lb[d]:
                        trial[d] = 2 * lb[d] - trial[d]
                    elif trial[d] > ub[d]:
                        trial[d] = 2 * ub[d] - trial[d]
                # Final clamp to be safe
                trial = np.clip(trial, lb, ub)

                # Evaluate
                trial_fit = func(trial)

                # Greedy selection
                if trial_fit < fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

        return (best_x, best_y)
