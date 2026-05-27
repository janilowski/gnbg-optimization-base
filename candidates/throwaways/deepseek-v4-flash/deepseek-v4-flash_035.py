import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact Differential Evolution (DE/rand/1/bin) 
# optimizer for black-box minimization. It is designed to be robust across 
# different dimensions and budget sizes, using only numpy and the Python 
# standard library.
# Search state: A population of candidate solutions (real vectors) and their 
# corresponding objective values. The best-found solution is tracked.
# Candidate generation: For each target individual, a mutant is created by adding 
# the scaled difference of two randomly selected distinct population vectors to 
# a third base vector. Crossover with the target vector produces a trial vector.
# Selection and replacement: Greedy selection: if the trial vector yields a lower 
# objective value, it replaces the target individual.
# Adaptation: No online adaptation of parameters. The scaling factor F and 
# crossover probability CR are fixed (F=0.8, CR=0.9), providing a good 
# default for many problems.
# Exploration mechanisms: The mutation operator with random base and difference 
# vectors encourages exploration of the search space, especially when the 
# population is diverse.
# Exploitation mechanisms: As the population converges, difference vectors 
# shrink, focusing the search around promising areas. Crossover also helps 
# preserve good components from successful solutions.
# Boundary handling: Trial vectors that exceed lower or upper bounds are clipped 
# to the bounds.
# Budget strategy: The population size is set to min(50, budget//10, 10*dim), 
# ensuring at least 3 individuals. The algorithm stops as soon as the 
# evaluation budget is exhausted, even in the middle of a generation.
# Closest known influences: Standard Differential Evolution (DE/rand/1/bin) 
# with fixed parameters.
# Novelty or unusual aspects: None; the implementation follows textbook DE 
# with a simple budget-aware population sizing.
# Failure modes: With very low budgets (<=2), it falls back to pure random 
# search. For extremely high-dimensional problems with a low budget, 
# performance may degrade due to insufficient population diversity.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the optimizer with a given evaluation budget and problem dimension.

        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimizer on the provided objective function.

        Parameters
        ----------
        func : callable
            The black-box objective function to minimize. It is expected to expose
            either `func.lower` / `func.upper` or `func.bounds.lb` / `func.bounds.ub`
            to define the search domain.

        Returns
        -------
        tuple
            (best_x, best_y) where best_x is a 1-D numpy array of the best solution
            found, and best_y is the corresponding scalar objective value.
        """
        # ----- Read domain bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective function must provide bounds via "
                                 "func.lower/upper or func.bounds.lb/ub")

        dim = self.dim
        budget = self.budget

        # ----- Fallback for very small budgets: pure random search -----
        if budget < 3:
            best_x = None
            best_y = np.inf
            for _ in range(budget):
                x = lower + np.random.rand(dim) * (upper - lower)
                y = func(x)
                if y < best_y:
                    best_x, best_y = x.copy(), y
            return best_x, best_y

        # ----- Population sizing -----
        # At least 3, at most 50, and never more than budget // 10 or 10*dim
        npop = max(3, min(50, budget // 10, 10 * dim))
        if npop > budget:
            npop = budget  # should not happen but safeguard

        # ----- Initialization -----
        pop = lower + np.random.rand(npop, dim) * (upper - lower)
        fitness = np.full(npop, np.inf)
        best_idx = 0
        for i in range(npop):
            val = func(pop[i])
            fitness[i] = val
            if val < fitness[best_idx]:
                best_idx = i
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        evals_used = npop
        remaining = budget - evals_used

        # ----- DE parameters -----
        F = 0.8       # scaling factor
        CR = 0.9      # crossover probability

        # ----- Main DE loop -----
        while remaining > 0:
            # Shuffle order of processing each generation to reduce bias
            order = np.random.permutation(npop)
            for i in order:
                # Pick three distinct random indices != i
                candidates = [j for j in range(npop) if j != i]
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                # Mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover
                cross_points = np.random.rand(dim) < CR
                # Ensure at least one component from mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                # Clip to bounds
                trial = np.clip(trial, lower, upper)

                # Evaluate
                trial_f = func(trial)
                evals_used += 1
                remaining -= 1

                # Greedy selection
                if trial_f < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_y:
                        best_y = trial_f
                        best_x = trial.copy()

                if remaining <= 0:
                    break

        return best_x, best_y
