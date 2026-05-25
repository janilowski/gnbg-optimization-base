import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implementation of a Differential Evolution (DE) algorithm for black-box minimization.
# Search state: A population of candidate vectors and their corresponding fitness values.
# Candidate generation: For each target vector, a mutant is created by adding the scaled difference of two random population vectors to a third. Binomial crossover combines the mutant with the target to produce a trial vector.
# Selection and replacement: Greedy selection – the trial vector replaces the target if it yields lower (better) fitness.
# Adaptation: Fixed scale factor F=0.8 and crossover rate CR=0.9. Population size is chosen based on dimension and budget.
# Exploration mechanisms: Random differences in mutation and high crossover probability promote exploration. Population diversity is maintained by random selection of base and difference vectors.
# Exploitation mechanisms: Greedy selection and tracking of the best solution encountered so far.
# Boundary handling: Trial vectors are clipped to the search bounds.
# Budget strategy: Each evaluation increments a counter; the algorithm stops when the budget is exhausted.
# Closest known influences: Classic Differential Evolution (DE/rand/1/bin).
# Novelty or unusual aspects: Simple, compact implementation with no adaptive parameter tuning or restarts.
# Failure modes: May converge prematurely on highly multimodal landscapes; fixed parameters may not suit all functions; limited population size can be problematic for high-dimensional problems with small budgets.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the algorithm with a maximum number of function evaluations (budget)
        and the problem dimensionality (dim).
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the Differential Evolution optimizer on the given function.

        Parameters
        ----------
        func : callable
            The objective function to minimize. Must provide either
            `func.lower` / `func.upper` or `func.bounds.lb` / `func.bounds.ub`.

        Returns
        -------
        best_x : numpy.ndarray
            The best solution found.
        best_y : float
            The corresponding objective value.
        """
        # ---------- Extract bounds ----------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot determine search bounds from the function object.")

        # ---------- Algorithm parameters ----------
        # Population size: at least 4, at most 50, scaled by dimension but capped by budget
        NP = max(4, min(50, int(self.dim * 10)))
        NP = min(NP, self.budget)                # cannot exceed total budget
        F = 0.8                                  # mutation factor
        CR = 0.9                                 # crossover probability

        # ---------- Initial population ----------
        pop = np.random.uniform(lb, ub, size=(NP, self.dim))
        pop_fitness = np.full(NP, np.inf)
        evals = 0

        for i in range(NP):
            if evals >= self.budget:
                break
            pop_fitness[i] = func(pop[i])
            evals += 1

        # Track best solution
        best_idx = np.argmin(pop_fitness)
        best_x = pop[best_idx].copy()
        best_y = pop_fitness[best_idx]

        # ---------- Main DE loop ----------
        while evals < self.budget:
            for i in range(NP):
                if evals >= self.budget:
                    break

                # Mutation: choose three distinct indices (a, b, c) all different from i
                indices = list(range(NP))
                indices.remove(i)
                a, b, c = np.random.choice(indices, size=3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover (ensure at least one dimension is crossed)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1

                # Greedy selection
                if trial_fitness < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
