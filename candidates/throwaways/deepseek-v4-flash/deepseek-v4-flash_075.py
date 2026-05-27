import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact implementation of Differential Evolution (DE) with rand/1/bin strategy for black-box minimization.
# Search state: A population of candidate solutions stored in array 'pop' with their objective values 'f_pop'. Best solution tracked.
# Candidate generation: For each target vector, generate a mutant vector using DE/rand/1: V = X_r1 + F*(X_r2 - X_r3). Then binomial crossover with probability CR produces a trial vector.
# Selection and replacement: Greedy selection: if trial vector yields lower or equal objective value, it replaces the target in the population.
# Adaptation: F and CR are fixed (F=0.8, CR=0.9) for simplicity; no adaptation.
# Exploration mechanisms: Mutation with differential variation provides exploration, crossover mixes dimensions.
# Exploitation mechanisms: Selection retains better solutions; over generations the population converges.
# Boundary handling: Clamping trial vectors to the bounds ([low, high]).
# Budget strategy: The total number of function evaluations is fixed to the provided budget. The population size is determined as NP = min(budget, max(4, 10*dim, budget//10)) to ensure multiple generations if possible.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None. Straightforward DE. Boundary handling via clamping.
# Failure modes: May converge prematurely on multimodal problems; low budget severely limits generations; fixed F/CR may not suit all landscapes; population size may be too small for high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.array(func.lower, dtype=float)
            upper = np.array(func.upper, dtype=float)
        else:
            lower = np.array(func.bounds.lb, dtype=float)
            upper = np.array(func.bounds.ub, dtype=float)

        # Ensure arrays are 1-D of length dim
        if lower.ndim == 0:
            lower = np.full(self.dim, lower)
            upper = np.full(self.dim, upper)
        elif lower.shape[0] != self.dim:
            lower = np.full(self.dim, lower.item())
            upper = np.full(self.dim, upper.item())

        budget = self.budget
        dim = self.dim

        # Population size: at least 4, at most budget, influenced by dimension
        NP = max(4, min(budget, 10 * dim, budget // 10))
        NP = min(NP, budget)

        # Initialise population uniformly in the domain
        pop = np.random.uniform(lower, upper, size=(NP, dim))
        f_pop = np.array([func(x) for x in pop])
        evals = NP

        # Track best solution
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_y = f_pop[best_idx]

        # DE parameters
        F = 0.8
        CR = 0.9

        # Main DE loop
        while evals < budget:
            for i in range(NP):
                # Choose three distinct random indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.array([
                    mutant[j] if (np.random.rand() < CR or j == j_rand) else pop[i, j]
                    for j in range(dim)
                ])

                # Boundary clamping
                trial = np.clip(trial, lower, upper)

                # Evaluation
                f_trial = func(trial)
                evals += 1

                # Greedy selection
                if f_trial <= f_pop[i]:
                    pop[i] = trial
                    f_pop[i] = f_trial
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()

                # Check budget after each evaluation
                if evals >= budget:
                    break

        return best_x, best_y
