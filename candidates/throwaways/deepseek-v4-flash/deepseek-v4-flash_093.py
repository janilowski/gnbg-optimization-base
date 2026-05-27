# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution with self-adaptive parameters (jDE variant) for GNBG minimization benchmark.
# Search state: Population of vectors with associated F and CR parameters, plus global best.
# Candidate generation: DE/rand/1 mutation and binomial crossover.
# Selection and replacement: One-to-one greedy selection between target and trial vectors.
# Adaptation: Each individual carries its own F (scale factor) and CR (crossover rate). After each generation,
#   parameters are regenerated with fixed probabilities (0.1 for F, 0.1 for CR) to new random values within [0.1,1.0] for F and [0,1] for CR.
# Exploration mechanisms: Mutation using difference vectors (three random individuals) and periodic parameter regeneration.
# Exploitation mechanisms: Crossover combines target vector with mutant, greedy selection retains improvements.
# Boundary handling: Reflection (bouncing) of coordinates that exceed bounds.
# Budget strategy: All evaluations are counted; the algorithm stops when the next batch would exceed the remaining budget.
# Closest known influences: jDE (Brest et al., 2006), classic DE/rand/1/bin.
# Novelty or unusual aspects: None – a straightforward implementation of a well‑known adaptive DE.
# Failure modes: May struggle on highly multimodal landscapes with small population size; may stagnate if population diversity is lost.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = max(4 * dim, 40)          # population size scales with dimension

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            # assume func.bounds exists as an object with lb and ub
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        # Ensure bounds are 1D arrays of correct dimension
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)

        pop_size = min(self.pop_size, self.budget)   # cannot have more than budget individuals
        # if budget too small to evaluate one full population, fall back to random search
        if pop_size < 1:
            # evaluate at most one point
            best_x = np.random.uniform(lb, ub)
            best_y = func(best_x)
            return best_x.tolist(), best_y

        # Initialisation
        pop = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        y = np.full(pop_size, np.inf)
        for i in range(pop_size):
            y[i] = func(pop[i])
        evals = pop_size

        # Best so far
        best_idx = np.argmin(y)
        best_x = pop[best_idx].copy()
        best_y = y[best_idx]

        # Self-adaptive parameters for each individual
        F = np.random.uniform(0.1, 1.0, size=pop_size)
        CR = np.random.uniform(0, 1, size=pop_size)

        # Main loop – process as many generations as possible
        # Each generation evaluates pop_size trial vectors
        while evals + pop_size <= self.budget:
            # For each target, generate trial vector, evaluate, replace if better
            for i in range(pop_size):
                # Mutation: DE/rand/1
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(idxs, size=3, replace=False)
                mutant = pop[a] + F[i] * (pop[b] - pop[c])

                # Binomial crossover
                j_rand = np.random.randint(0, self.dim)
                trial = np.where(
                    np.random.rand(self.dim) < CR[i],
                    mutant,
                    pop[i]
                )
                # Ensure at least one component from mutant
                trial[j_rand] = mutant[j_rand]

                # Boundary handling – reflect
                low_viol = trial < lb
                high_viol = trial > ub
                trial[low_viol] = lb[low_viol] + (lb[low_viol] - trial[low_viol])
                trial[high_viol] = ub[high_viol] - (trial[high_viol] - ub[high_viol])
                # Clamp in case reflection still out of bounds (unlikely but safe)
                trial = np.clip(trial, lb, ub)

                # Evaluate
                trial_y = func(trial)
                evals += 1

                # Greedy selection
                if trial_y < y[i]:
                    pop[i] = trial
                    y[i] = trial_y
                    if trial_y < best_y:
                        best_x = trial.copy()
                        best_y = trial_y
                    # Parameters survive (success)
                else:
                    # Adaptation: regenerate F and CR with small probability
                    if np.random.rand() < 0.1:
                        F[i] = np.random.uniform(0.1, 1.0)
                    if np.random.rand() < 0.1:
                        CR[i] = np.random.uniform(0, 1)

            # End of generation – check budget for next generation

        # If budget remains but not enough for a full generation, evaluate remaining one‑by‑one
        remaining = self.budget - evals
        while remaining > 0:
            # generate random trial (or use best and perturb)
            # Simple approach: generate one new random point
            trial = np.random.uniform(lb, ub)
            trial_y = func(trial)
            evals += 1
            remaining -= 1
            if trial_y < best_y:
                best_x = trial.copy()
                best_y = trial_y

        # Return best_x as a list (common for benchmarking), best_y as float
        return best_x.tolist(), best_y
