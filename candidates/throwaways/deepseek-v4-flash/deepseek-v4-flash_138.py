# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE/rand/1/bin) with fixed parameters for black-box minimization.
# Search state: A population of candidate solutions (vectors) uniformly distributed within the search space.
# Candidate generation: For each target individual, a mutant vector is created by adding the scaled difference of two random population members to a third random member. Then binomial crossover combines the mutant with the target to form a trial vector.
# Selection and replacement: The trial replaces the target if it yields a lower (better) objective value; otherwise the target is retained. No elitism beyond this per-individual greedy selection.
# Adaptation: No adaptation of control parameters (F, CR) – static values chosen for robustness.
# Exploration mechanisms: The differential mutation step induces exploration by searching along random directions; the crossover mixes components from mutant and target.
# Exploitation mechanisms: The greedy selection allows the population to converge gradually as fitter individuals are retained and used in mutation.
# Boundary handling: Trial vectors that violate bounds are clipped to the nearest bound value.
# Budget strategy: The initial population is fully evaluated, then generations are run until the remaining budget is exhausted; the last generation may be partial. Total evaluations never exceed the provided budget.
# Closest known influences: Standard DE/rand/1/bin (Storn & Price, 1997).
# Novelty or unusual aspects: None – this is a textbook implementation.
# Failure modes: May struggle with highly multimodal or deceptive landscapes due to fixed parameters; can converge prematurely if population diversity collapses; static F and CR may not suit all problem types.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initialize with total evaluation budget and problem dimension.
        """
        self.budget = int(budget)
        self.dim = int(dim)

        # Control parameters for DE
        self.F = 0.5          # differential weight
        self.CR = 0.9         # crossover probability

        # Population size: at least 10, at most budget//2 to allow generations
        self.pop_size = min(max(10, self.dim * 4), self.budget // 2)
        if self.pop_size < 4:
            self.pop_size = 4  # need at least 4 for mutation to work

    def __call__(self, func):
        """
        Run the optimizer and return the best found point and its objective value.
        """
        # Extract bounds from the function object (handles both forms)
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        dim = self.dim

        # Initialise population uniformly in the box
        pop = np.random.uniform(lb, ub, size=(self.pop_size, dim))
        pop_y = np.full(self.pop_size, np.inf)
        best_x = None
        best_y = np.inf
        evals = 0

        # Evaluate initial population
        for i in range(self.pop_size):
            y = func(pop[i])
            evals += 1
            pop_y[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Main DE loop
        while evals < self.budget:
            # For each target vector, generate a trial
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break

                # Choose three distinct random indices different from i
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                r = np.random.choice(candidates, size=3, replace=False)
                a, b, c = r[0], r[1], r[2]

                # Mutation: mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = pop[a] + self.F * (pop[b] - pop[c])

                # Crossover: binomial (uniform) crossover
                j_rand = np.random.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i, j]

                # Boundary handling – clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_y = func(trial)
                evals += 1

                # Greedy selection (minimization)
                if trial_y < pop_y[i]:
                    pop[i] = trial
                    pop_y[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()

        return best_x, best_y
