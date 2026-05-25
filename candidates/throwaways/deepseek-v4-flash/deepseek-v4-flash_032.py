import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE) with the classical rand/1/bin strategy.
# Search state: A population of candidate solutions and their corresponding objective
#   values; the best-found solution is tracked.
# Candidate generation: For each population vector (target), a mutant vector is created
#   by adding the scaled difference of two random distinct population vectors to a third
#   random vector (mutation). A trial vector is formed by binomial crossover between the
#   target and mutant with probability CR.
# Selection and replacement: One-to-one greedy selection: if the trial vector has a lower
#   (minimization) objective value than the target, it replaces the target in the population.
# Adaptation: Fixed parameters – population size = max(10*dim, 20) clipped to budget/2,
#   scaling factor F = 0.5, crossover rate CR = 0.9.
# Exploration mechanisms: Mutation uses random distinct individuals, promoting diversity;
#   crossover mixes dimensions.
# Exploitation mechanisms: The greedy selection drives the population toward better regions
#   over generations.
# Boundary handling: Trial vectors that fall outside the box bounds are clipped to the
#   nearest bound.
# Budget strategy: The algorithm stops as soon as the number of function evaluations reaches
#   the given budget. The population is initialised and evaluated (popsize evaluations), then
#   each generation requires popsize evaluations (one per trial).
# Closest known influences: Classic differential evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None – a straightforward implementation.
# Failure modes: Can stagnate or converge prematurely if F and CR are poorly suited to the
#   problem; high dimensions or limited budgets may reduce effectiveness; clipping disturbs
#   the mutation difference vectors and may cause loss of diversity.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the optimizer with a fixed evaluation budget and dimensionality.

        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # DE parameters (fixed, but scaled to budget/dim)
        # Population size: heuristic – at least 20, at most budget/2, scaled with dimension
        self.popsize = min(max(10 * dim, 20), budget // 2)
        if self.popsize < 5:  # safety for very low budget
            self.popsize = max(3, budget // 3)

        self.F = 0.5   # mutation scaling factor
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        """
        Run the optimization.

        Parameters
        ----------
        func : callable
            The objective function. Must provide lower/upper or bounds attributes.
            func.lower, func.upper   (or) func.bounds.lb, func.bounds.ub

        Returns
        -------
        best_x : np.ndarray (dim,)
        best_y : float
        """
        # ---- read bounds ----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot determine bounds from the function.")
        lb = lower
        ub = upper

        dim = self.dim
        popsize = self.popsize
        budget = self.budget

        # ---- initialisation ----
        # population: array of shape (popsize, dim)
        pop = lb + (ub - lb) * np.random.rand(popsize, dim)
        # evaluate
        evals = 0
        fitness = np.full(popsize, np.inf)
        for i in range(popsize):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1

        # best so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # ---- main DE loop ----
        while evals < budget:
            # For each target vector, generate a trial vector
            for i in range(popsize):
                if evals >= budget:
                    break

                # Choose three distinct indices different from i
                candidates = list(range(popsize))
                candidates.remove(i)
                idxs = np.random.choice(candidates, size=3, replace=False)
                a, b, c = idxs

                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                mutant = pop[a] + self.F * (pop[b] - pop[c])

                # Binomial crossover: jrand ensures at least one dimension from mutant
                trial = pop[i].copy()
                jrand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < self.CR or j == jrand:
                        trial[j] = mutant[j]

                # Clip trial to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate
                trial_fitness = func(trial)
                evals += 1

                # Selection: greedy one-to-one
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    # update global best
                    if trial_fitness < best_y:
                        best_x = trial.copy()
                        best_y = trial_fitness

            # If budget is exhausted after evaluating a generation, break
            if evals >= budget:
                break

        return best_x, best_y
