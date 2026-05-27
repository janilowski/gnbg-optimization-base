import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a simple Differential Evolution (DE) algorithm with rand/1/bin strategy. It maintains a population of candidate solutions, generates trial vectors by mutating a base vector with scaled differences and binomial crossover, and replaces parents if offspring are better. It is designed for black-box minimization.
# Search state: A population of NP individuals (real vectors) of dimension D, and their corresponding function values. The global best solution found so far is tracked separately.
# Candidate generation: For each target individual, three distinct random individuals (excluding the target) are chosen. A mutant vector is created as base + F * (difference1 - difference2). Then a trial vector is formed by binomial crossover with the target with probability CR per dimension.
# Selection and replacement: Greedy selection: after evaluating the trial (if budget permits), if its objective value is less than or equal to the target's, the target is replaced by the trial.
# Adaptation: No adaptation; fixed parameters F=0.8, CR=0.9.
# Exploration mechanisms: Mutation using random differences provides diversity. Higher F and crossover (CR) encourage exploration. Random selection of base and difference vectors prevents convergence to a single direction prematurely (rand/1).
# Exploitation mechanisms: Selection pressure favoring better solutions gradually focuses the population. The greedy replacement maintains best found solutions.
# Boundary handling: Candidate vectors generated outside bounds are clipped to the lower or upper bound.
# Budget strategy: The population size NP is set based on budget: NP = max(5, min(10*dim, int(budget/10))). This ensures a reasonable population size that leaves enough evaluations for generations. The main loop stops when the remaining budget is insufficient for a full generation, but individual trials are evaluated one by one and stopped when budget runs out.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1995).
# Novelty or unusual aspects: None; this is a vanilla implementation.
# Failure modes: May stagnate on highly multimodal landscapes if population size is too small or if F/CR are unsuitable. For very high-dimensional problems with limited budget, the population may not cover the search space effectively. Also, because parameters are fixed, it may fail on problems requiring different mutation scales.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Read bounds from the function object
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb, dtype=float)
                ub = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError as e:
                raise AttributeError(
                    "Function must provide either .lower/.upper or .bounds.lb/.bounds.ub"
                ) from e

        # Ensure bounds are 1-D arrays of length dim
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)
        lb = lb.astype(float)
        ub = ub.astype(float)

        # Population size heuristics
        NP = max(5, min(10 * self.dim, int(self.budget / 10)))
        # Reduce NP if budget is extremely small (should not happen normally)
        NP = min(NP, self.budget // 2)
        NP = max(NP, 4)  # at least 4 for mutation

        # Initialize population
        rng = np.random.default_rng()
        pop = rng.uniform(lb, ub, size=(NP, self.dim))
        pop_f = np.full(NP, np.inf)

        # Evaluate initial population
        evals = 0
        for i in range(NP):
            if evals >= self.budget:
                break
            pop_f[i] = func(pop[i])
            evals += 1

        # Track global best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_y = pop_f[best_idx]

        # DE parameters
        F = 0.8
        CR = 0.9

        # Main loop: iterate over generations, but stop when budget exhausted
        generation = 0
        while evals < self.budget:
            generation += 1
            for i in range(NP):
                if evals >= self.budget:
                    break

                # Select three distinct random indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                r = rng.choice(candidates, size=3, replace=False)
                a, b, c = r[0], r[1], r[2]

                # Mutation
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                j_rand = rng.integers(self.dim)
                trial = np.empty(self.dim)
                for d in range(self.dim):
                    if rng.random() < CR or d == j_rand:
                        trial[d] = mutant[d]
                    else:
                        trial[d] = pop[i, d]

                # Boundary clipping
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_f = func(trial)
                evals += 1

                # Greedy selection
                if trial_f <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    # Update global best if improved
                    if trial_f < best_y:
                        best_y = trial_f
                        best_x = trial.copy()

        return best_x, best_y
