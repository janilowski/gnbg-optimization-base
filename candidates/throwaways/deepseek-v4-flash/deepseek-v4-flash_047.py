# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple Differential Evolution (DE/rand/1/bin) minimizer adapted for black-box optimization. It maintains a fixed-size population of candidate solutions, generates trial vectors by mutation and crossover, and replaces inferior parents.
# Search state: A population (numpy array) of shape (pop_size, dim) plus an array of corresponding objective values. The best solution seen so far is stored explicitly.
# Candidate generation: For each parent, three distinct individuals are randomly chosen from the population (excluding the parent). The scaled difference of two of them is added to the third (mutation), then binomial crossover with the parent produces a trial vector.
# Selection and replacement: Greedy selection – the trial vector replaces the parent if its objective value is no worse (minimization). The best overall solution is updated after each trial evaluation.
# Adaptation: None – scaling factor F and crossover probability CR are fixed at 0.8 and 0.9 respectively.
# Exploration mechanisms: Mutation using random differential vectors provides diversity; binomial crossover mixes parent and mutant components.
# Exploitation mechanisms: Greedy replacement favors better solutions; the population gradually converges. The best-so-far solution is retained and returned.
# Boundary handling: Trial vectors are clamped to the lower and upper bounds after generation.
# Budget strategy: The initial population uses exactly pop_size evaluations. After that, each trial consumes one evaluation. The loop stops when the total number of evaluations reaches the budget.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997), specifically DE/rand/1/bin.
# Novelty or unusual aspects: The population size is chosen automatically as a function of dimension and budget, never exceeding budget. Bound handling via simple clamping. No adaptation of control parameters.
# Failure modes: On very low budgets or high-dimensional problems, the population may be small, limiting exploration. Fixed F and CR may perform poorly on some landscapes with sharp ridges or variable sensitivity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Prepare the optimizer for a given budget of function evaluations
        and problem dimension.
        """
        self.budget = budget
        self.dim = dim
        # Default control parameters – tuned for a reasonable balance
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        """
        Run the optimizer on callable *func* which returns a scalar for
        minimization.  Returns (best_x, best_y) using at most *budget* evaluations.
        """
        # ----- read bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.atleast_1d(np.asarray(func.lower, dtype=float))
            ub = np.atleast_1d(np.asarray(func.upper, dtype=float))
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.atleast_1d(np.asarray(func.bounds.lb, dtype=float))
            ub = np.atleast_1d(np.asarray(func.bounds.ub, dtype=float))
        else:
            raise ValueError("Cannot determine bounds from func object")
        # Ensure bounds are 1‑D arrays of length dim
        if lb.ndim == 0:
            lb = np.full(self.dim, lb.item())
        if ub.ndim == 0:
            ub = np.full(self.dim, ub.item())
        lb = lb.astype(float).reshape(self.dim)
        ub = ub.astype(float).reshape(self.dim)
        # verify order
        if np.any(lb > ub):
            lb, ub = ub, lb

        # ----- population size -----
        # reasonable heuristic: at least 4, at most 10*dim, but never exceed budget/4
        pop_size = max(4, min(10 * self.dim, self.budget // 4))
        # ensure we have enough budget for initial population + at least one generation
        if pop_size > self.budget:
            pop_size = self.budget  # degenerate case: only initial evaluations

        # ----- initialisation -----
        pop = lb + (ub - lb) * np.random.rand(pop_size, self.dim)
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = np.empty(self.dim)
        best_y = np.inf

        for i in range(pop_size):
            evals += 1
            val = func(pop[i])
            fitness[i] = val
            if val < best_y:
                best_y = val
                best_x = pop[i].copy()

        # ----- main DE loop -----
        while evals < self.budget:
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # choose three distinct indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                idx = np.random.choice(candidates, 3, replace=False)
                a, b, c = idx[0], idx[1], idx[2]
                # mutation
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                # binomial crossover
                cross_points = np.random.rand(self.dim) < self.CR
                # ensure at least one cross point
                if not np.any(cross_points):
                    cross_points[np.random.randint(self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                # boundary handling: clamp
                trial = np.clip(trial, lb, ub)
                # evaluate
                evals += 1
                val = func(trial)
                # greedy selection
                if val <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_y:
                        best_y = val
                        best_x = trial.copy()

        return best_x, best_y
