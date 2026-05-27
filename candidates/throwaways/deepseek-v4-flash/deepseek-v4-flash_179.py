# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a standard Differential Evolution (DE/rand/1/bin) algorithm for black-box minimization.
# Search state: A fixed-size population of candidate solutions (real vectors) plus the current best solution.
# Candidate generation: For each target individual, three distinct random individuals are chosen. A mutant vector is
#   created as base + F * (difference vector). Binomial crossover with the target produces a trial vector.
# Selection and replacement: The trial vector is compared to the target; the better one (by function value) replaces
#   the target in the population.
# Adaptation: None – the mutation factor F and crossover rate CR are static (defaults: F=0.8, CR=0.9).
# Exploration mechanisms: The differential mutation introduces diverse directions; crossover can recombine features;
#   random index selection ensures diversity.
# Exploitation mechanisms: Greedy selection retains improvements; the population gradually converges to promising
#   regions.
# Boundary handling: Trial vectors are clamped component-wise to the box constraints [lower, upper].
# Budget strategy: The algorithm runs generations until the number of function evaluations reaches the allowed budget.
#   Evaluation count is tracked explicitly.
# Closest known influences: DE/rand/1/bin (Storn & Price, 1997).
# Novelty or unusual aspects: None – intentionally classic.
# Failure modes: May stagnate in rugged landscapes; static parameters may be suboptimal for some problems; can
#   prematurely converge if population diversity is lost quickly.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Differential Evolution (DE/rand/1/bin) with fixed parameters.

    Minimizes a given black-box function subject to box constraints.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Parameters of DE
        self.F = 0.8
        self.CR = 0.9
        # Population size heuristic
        self.pop_size = max(5, min(10 * dim, 50))
        # Ensure we don't allocate more evaluations than budget for initial population
        self.pop_size = min(self.pop_size, budget)

    def __call__(self, func):
        # ---- read bounds ----
        try:
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lower = np.asarray(func.bounds.lb, dtype=float)
                upper = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Cannot determine box bounds from function object.")
        if lower.ndim == 0:
            lower = np.full(self.dim, lower)
            upper = np.full(self.dim, upper)
        # ---- initialisation ----
        pop = np.random.uniform(lower, upper, (self.pop_size, self.dim))
        evals = 0
        best_x = None
        best_y = np.inf
        # Evaluate initial population
        fitness = np.empty(self.pop_size)
        for i in range(self.pop_size):
            y = func(pop[i])
            fitness[i] = y
            evals += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()
        if evals >= self.budget:
            return best_x, best_y

        # ---- main DE loop ----
        while evals < self.budget:
            # Shuffle population order each generation (optional but common)
            indices = np.random.permutation(self.pop_size)
            for idx in indices:
                if evals >= self.budget:
                    break

                # pick three distinct random indices different from idx
                candidates = [j for j in range(self.pop_size) if j != idx]
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # mutation
                mutant = pop[a] + self.F * (pop[b] - pop[c])

                # binomial crossover
                trial = pop[idx].copy()
                # random dimension to ensure at least one change
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # boundary clamping
                trial = np.clip(trial, lower, upper)

                # evaluation & selection
                y_trial = func(trial)
                evals += 1
                if y_trial < fitness[idx]:
                    pop[idx] = trial
                    fitness[idx] = y_trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()

        return best_x, best_y
