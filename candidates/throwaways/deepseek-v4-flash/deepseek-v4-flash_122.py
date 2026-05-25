import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE/rand/1/bin) with reflection boundary handling.
# Search state: A population of candidate solutions (vectors) of size NP = max(4, min(50, 10*dim)),
#              stored as a 2D numpy array. The best solution found so far is tracked.
# Candidate generation: For each target vector, a mutant vector is created by adding the weighted
#                       difference of two random population members to a base member (rand/1).
#                       Crossover (binomial) blends the mutant with the target to produce a trial vector.
# Selection and replacement: The trial vector replaces the target if its fitness is better.
# Adaptation: The mutation scaling factor F and crossover probability CR are fixed (F=0.8, CR=0.9).
# Exploration mechanisms: High crossover probability and random differential mutation encourage
#                         diverse exploration during early generations.
# Exploitation mechanisms: As the population converges, the differences shrink, focusing search
#                          around promising areas. The best solution is maintained.
# Boundary handling: Reflective repair: components outside [lb, ub] are reflected inward.
# Budget strategy: The algorithm evaluates the initial population (NP evaluations) then iterates
#                  generations, evaluating at most one trial per individual per generation until
#                  the budget is exhausted.
# Closest known influences: Classic DE algorithm (Storn & Price, 1997).
# Novelty or unusual aspects: None; a textbook implementation with a conservative strategy for
#                             handling small budgets (population size capped relative to dim).
# Failure modes: May stagnate on highly multimodal or deceptive landscapes, especially with small
#                population sizes. Fixed parameters may not suit all problems.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initialize the differential evolution optimizer.
        :param budget: maximum number of function evaluations allowed.
        :param dim: dimensionality of the problem.
        """
        self.budget = int(budget)
        self.dim = int(dim)
        # Population size: at least 4, at most 50, and roughly 10*dim.
        self.NP = max(4, min(50, 10 * self.dim))
        self.F = 0.8          # mutation scaling factor
        self.CR = 0.9         # crossover probability

    def __call__(self, func):
        """
        Run the optimizer on the given objective function.
        :param func: objective function object with bounds (func.lower / func.upper
                     or func.bounds.lb / func.bounds.ub) and __call__.
        :return: tuple (best_x, best_y) where best_x is the best solution vector
                 and best_y is its objective value.
        """
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.atleast_1d(np.asarray(func.lower, dtype=float))
            ub = np.atleast_1d(np.asarray(func.upper, dtype=float))
        else:
            lb = np.atleast_1d(np.asarray(func.bounds.lb, dtype=float))
            ub = np.atleast_1d(np.asarray(func.bounds.ub, dtype=float))

        # Ensure bounds are 1-D arrays of correct dimension
        lb = lb.ravel().astype(float)
        ub = ub.ravel().astype(float)
        if len(lb) == 1:
            lb = np.full(self.dim, lb[0])
            ub = np.full(self.dim, ub[0])

        # Initialize population uniformly within bounds
        population = lb + (ub - lb) * np.random.rand(self.NP, self.dim)
        # Evaluate population
        fitness = np.full(self.NP, np.inf)
        evals = 0
        for i in range(self.NP):
            if evals >= self.budget:
                break
            fitness[i] = func(population[i])
            evals += 1

        # Track best ever
        best_idx = np.argmin(fitness[:evals])
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]

        # Main evolutionary loop
        while evals < self.budget:
            for i in range(self.NP):
                if evals >= self.budget:
                    break
                # Choose three distinct random indices different from i
                candidates = [j for j in range(self.NP) if j != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Mutation: DE/rand/1
                mutant = population[a] + self.F * (population[b] - population[c])

                # Binomial crossover
                cross_points = np.random.rand(self.dim) < self.CR
                # Ensure at least one component is inherited from mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Reflective boundary handling
                trial = self._reflect(trial, lb, ub)

                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y

    @staticmethod
    def _reflect(x, lb, ub):
        """Reflect components outside [lb, ub] back into the interval."""
        # Reflect below lower bound
        below = x < lb
        x[below] = lb[below] + (lb[below] - x[below])
        # Reflect above upper bound
        above = x > ub
        x[above] = ub[above] - (x[above] - ub[above])
        # If still out (possible after reflection due to multiple reflections), clip
        np.clip(x, lb, ub, out=x)
        return x
