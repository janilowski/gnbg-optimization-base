# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution with self-adaptive control parameters (jDE variant)
# for black-box minimization. A population of candidate solutions is evolved by
# mutation, crossover, and greedy selection. The scale factor F and crossover
# rate CR are individually adapted per solution using a probabilistic resetting
# scheme.
# Search state: a population of N vectors (each dim-dimensional) and their
# objective values; for each individual also a scalar F (mutation factor) and
# CR (crossover probability).
# Candidate generation: for each parent index i, three distinct random population
# members a, b, c (all different from i) are selected. A mutant vector is formed
# as v = pop[a] + F_i * (pop[b] - pop[c]). Binomial crossover with CR_i then
# combines v with the parent to form a trial vector.
# Selection and replacement: the trial vector is evaluated and replaces the parent
# if its objective value is lower (minimization) or equal. The associated F_i and
# CR_i are retained only if the trial was successful; otherwise they are reset to
# default (0.5 and 0.9 respectively) for the next generation.
# Adaptation: with probability tau1=0.1, F_i is replaced by a new value drawn
# uniformly from [0.1, 1.0]; with probability tau2=0.1, CR_i is replaced by a
# value from [0.0, 1.0]. Successful values are kept; unsuccessful ones are
# reverted to defaults (0.5 / 0.9). This promotes adaptation of the algorithmic
# parameters to the problem landscape.
# Exploration mechanisms: DE/rand/1 mutation uses differences between random
# population members, maintaining diversity. Self-adaptive F values allow the
# step size to vary between individuals and over time.
# Exploitation mechanisms: greedy selection always keeps the better solution,
# focusing effort on promising regions. The recombination (crossover) also
# exploits building blocks from successful parents. As the population converges,
# the difference vectors shrink, reducing the step size.
# Boundary handling: each coordinate of the trial vector is clipped to the
# variable bounds [lb, ub] provided by the test function.
# Budget strategy: the population size N is set as min(max(4, dim*3),
# max(2, budget//2), 50) to ensure at least one generation of evolution when
# budget allows, but never exceeds budget. Each function evaluation is counted,
# and the algorithm stops as soon as the budget is exhausted.
# Closest known influences: jDE (Brest et al., 2006), a classic self-adaptive
# differential evolution algorithm.
# Novelty or unusual aspects: none – a straightforward implementation of jDE.
# Failure modes: on very low budgets (e.g., <10 evaluations) the algorithm degrades
# to random search (initial population only). On highly multimodal landscapes
# the population may prematurely converge if N is too small. Self-adaptation
# can occasionally oscillate, but generally robust.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Self-adaptive Differential Evolution (jDE) for black-box minimization."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        budget = self.budget
        dim = self.dim

        # Determine variable bounds from the test function.
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # Some benchmarks store bounds as a Bounds object with .lb and .ub
            bounds = func.bounds
            lb = np.asarray(bounds.lb, dtype=float)
            ub = np.asarray(bounds.ub, dtype=float)
        else:
            # Fallback: assume a common interface
            lb = np.full(dim, -5.0)
            ub = np.full(dim,  5.0)

        # Population size: ensure at least 4 for DE, but never exceed budget.
        # Also limit to a reasonable size for efficiency.
        N = min(max(4, int(dim * 3)), max(2, budget // 2), 50)
        N = min(N, budget)  # cannot exceed total budget
        if N < 4:  # budget extremely small – just random search
            N = budget

        # Initialize population uniformly within bounds.
        pop = np.random.uniform(lb, ub, size=(N, dim))
        pop_f = np.full(N, np.inf)
        # Evaluate initial population
        evals = 0
        for i in range(N):
            if evals >= budget:
                break
            pop_f[i] = func(pop[i])
            evals += 1

        # Identify best so far
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_y = pop_f[best_idx]

        # If no evaluations left, return
        if evals >= budget:
            return best_x, best_y

        # Initialize control parameters for each individual
        # F (scale factor) and CR (crossover rate)
        F = np.full(N, 0.5)
        CR = np.full(N, 0.9)

        # Adaptation probabilities
        tau1 = 0.1
        tau2 = 0.1

        # Main evolution loop
        remaining = budget - evals
        generation = 0
        while remaining > 0:
            # Iterate through each population member
            for i in range(N):
                if remaining <= 0:
                    break

                # --- Self-adaptation of F and CR ---
                # Reset with small probability
                new_F = F[i]
                new_CR = CR[i]
                if np.random.rand() < tau1:
                    new_F = np.random.uniform(0.1, 1.0)
                if np.random.rand() < tau2:
                    new_CR = np.random.uniform(0.0, 1.0)

                # --- Mutation (DE/rand/1) ---
                # Select three distinct random indices different from i
                indices = list(range(N))
                indices.remove(i)
                a, b, c = np.random.choice(indices, size=3, replace=False)
                mutant = pop[a] + new_F * (pop[b] - pop[c])

                # --- Binomial crossover ---
                # Choose a random dimension to guarantee at least one change
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < new_CR,
                                 mutant,
                                 pop[i])
                # Ensure the j_rand-th dimension always comes from the mutant
                trial[j_rand] = mutant[j_rand]

                # --- Boundary handling (clipping) ---
                trial = np.clip(trial, lb, ub)

                # Evaluate trial if budget left
                trial_f = func(trial)
                remaining -= 1

                # --- Greedy selection ---
                if trial_f <= pop_f[i]:
                    # Accept trial
                    pop[i] = trial
                    pop_f[i] = trial_f
                    # Keep the new F and CR only if successful
                    F[i] = new_F
                    CR[i] = new_CR
                else:
                    # Revert to default values for next generation
                    F[i] = 0.5
                    CR[i] = 0.9

                # Update global best
                if trial_f < best_y:
                    best_y = trial_f
                    best_x = trial.copy()

            # End of generation
            generation += 1

        return best_x, best_y
