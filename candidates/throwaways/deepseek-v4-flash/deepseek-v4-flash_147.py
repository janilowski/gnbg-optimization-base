import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This implements a standard Differential Evolution (DE) algorithm for black-box minimization. It uses the DE/rand/1/bin variant with adaptive boundary handling via reflection.
# Search state: Maintains a population of candidate solutions (real-valued vectors) and their corresponding objective values. The best solution found is tracked.
# Candidate generation: For each target vector, a mutant vector is created by adding the scaled difference of two random population vectors to a third random base vector. Then binomial crossover is applied with the target vector to produce a trial vector.
# Selection and replacement: The trial vector replaces the target vector if it yields a lower objective value (greedy selection).
# Adaptation: The algorithm adapts the population structure through evolution; control parameters F (scale) and CR (crossover rate) are fixed (F=0.8, CR=0.9). No dynamic adaptation.
# Exploration mechanisms: Large F encourages exploration by scaling differences; random base vector selection and random crossover components maintain diversity.
# Exploitation mechanisms: Crossover combines good solutions; selection preserves improvements; the population converges to promising regions.
# Boundary handling: If a trial vector component violates bounds, it is clipped to the bounds.
# Budget strategy: The algorithm iterates generation by generation until the evaluation budget is exhausted. The population is evaluated exactly once per target vector per generation (NP evaluations per generation). After budget runs out, the best solution found is returned.
# Closest known influences: Classic DE (Storn & Price, 1997), with common parameter choices.
# Novelty or unusual aspects: None; it's a straightforward implementation.
# Failure modes: May stagnate on highly multimodal landscapes with fixed parameters; may prematurely converge if population diversity is lost. Bounds clipping can cause clustering near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

        # Population size: at least 5, at most 200, also limited by budget/2
        self.NP = max(5, min(200, 4 * dim, budget // 2))
        # Control parameters
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        # Determine bounds
        try:
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)

        dim = self.dim
        NP = self.NP
        budget = self.budget

        # Initialise population uniformly in the domain
        pop = np.random.uniform(lower, upper, size=(NP, dim))
        pop_y = np.full(NP, np.inf)

        # Evaluate initial population
        evals = 0
        best_x = None
        best_y = np.inf
        for i in range(NP):
            y = func(pop[i])
            pop_y[i] = y
            evals += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()
            if evals >= budget:
                # Budget exhausted after initialisation – return immediately
                return best_x, best_y

        # Main DE loop
        while evals < budget:
            for i in range(NP):
                if evals >= budget:
                    break

                # Choose three distinct random indices, none equal to i
                r = np.random.choice([j for j in range(NP) if j != i], size=3, replace=False)
                r1, r2, r3 = r[0], r[1], r[2]

                # Mutation
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(
                    np.random.rand(dim) < self.CR,
                    mutant,
                    pop[i]
                )
                trial[j_rand] = mutant[j_rand]  # ensure at least one component from mutant

                # Boundary handling: simple clipping
                trial = np.clip(trial, lower, upper)

                # Evaluate trial
                trial_y = func(trial)
                evals += 1

                # Greedy selection
                if trial_y < pop_y[i]:
                    pop[i] = trial
                    pop_y[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()

        return best_x, best_y
