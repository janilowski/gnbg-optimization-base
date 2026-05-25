import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a self-adaptive Differential Evolution (jDE variant)
#          for black-box minimization on the GNBG benchmark.
# Search state: A population of NP candidate vectors, each accompanied by its own
#              mutation factor F and crossover rate CR. The best-so-far solution
#              and its fitness are tracked.
# Candidate generation: For each population member, a mutant vector is created using
#                      DE/rand/1: v = r1 + F * (r2 - r3), where r1, r2, r3 are
#                      distinct random members (not the current index). Then a
#                      binomial crossover with per‑individual CR produces the trial.
# Selection and replacement: Greedy one‑to‑one replacement: if the trial’s fitness is
#                           no worse than the current member’s, it replaces the member.
# Adaptation: Before each trial generation, F and CR are updated with probabilities
#             tau1 = tau2 = 0.1. New F is drawn uniformly from [0.1, 1]; new CR from
#             [0, 1]. These values are inherited by offspring if replacement occurs.
# Exploration mechanisms: Random differential mutation, high crossover rates when CR
#                        is high, and frequent parameter resampling encourage exploration.
# Exploitation mechanisms: Greedy selection, decreasing population diversity over time,
#                          and low crossover rates when CR is low focus on local refinement.
# Boundary handling: Reflective boundary: components that exceed bounds are reflected
#                   back inside; if reflection still leaves them outside, they are clipped.
# Budget strategy: The initial population is evaluated once (NP evaluations). Then,
#                 one trial is generated and evaluated per iteration until the budget is
#                 exhausted. Each evaluation counts exactly one.
# Closest known influences: jDE (Brest et al., 2006) for self‑adaptation; classic
#                          DE/rand/1/bin by Storn and Price.
# Novelty or unusual aspects: Uses a steady‑state (iterative) replacement scheme instead
#                            of the usual generational update; this is combined with
#                            jDE parameter inheritance.
# Failure modes: Premature convergence on multimodal problems if population size is too
#               small; stagnation in high‑dimensional landscapes when NP is insufficient;
#               poor performance on extremely narrow basins due to reflection near bounds.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds from the function object
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb, dtype=float)
                ub = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Cannot read bounds from func")

        # Population size: at least 4, at most budget//2, scaled with dimension
        NP = max(4, min(2 * self.dim, self.budget // 2))
        # If budget is too small for even one generation of DE, fall back to random search
        if self.budget < NP + 1:
            # Random uniform sampling
            best_x = None
            best_y = np.inf
            for _ in range(self.budget):
                x = lb + (ub - lb) * np.random.rand(self.dim)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # Initialize population and their fitness
        pop = lb + (ub - lb) * np.random.rand(NP, self.dim)
        fit = np.full(NP, np.inf)
        for i in range(NP):
            fit[i] = func(pop[i])
        evals = NP

        # Self-adaptive parameters per individual: F in [0.1,1], CR in [0,1]
        F = 0.1 + 0.9 * np.random.rand(NP)
        CR = np.random.rand(NP)

        # Adaptation probabilities (jDE)
        tau1 = 0.1
        tau2 = 0.1

        # Steady-state DE loop: generate trial for each member cyclically
        i = 0
        while evals < self.budget:
            # Update F and CR for the current individual (jDE adaptation)
            if np.random.rand() < tau1:
                F[i] = 0.1 + 0.9 * np.random.rand()
            if np.random.rand() < tau2:
                CR[i] = np.random.rand()

            # Choose three distinct random indices different from i
            candidates = list(range(NP))
            candidates.remove(i)
            indices = np.random.choice(candidates, size=3, replace=False)
            r1, r2, r3 = indices

            # Mutation: DE/rand/1
            mutant = pop[r1] + F[i] * (pop[r2] - pop[r3])

            # Binomial crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < CR[i] or j == j_rand:
                    trial[j] = mutant[j]

            # Reflective boundary handling
            for j in range(self.dim):
                if trial[j] < lb[j]:
                    trial[j] = lb[j] + (lb[j] - trial[j])
                    if trial[j] > ub[j]:  # if reflection overshoots, clip
                        trial[j] = ub[j]
                elif trial[j] > ub[j]:
                    trial[j] = ub[j] - (trial[j] - ub[j])
                    if trial[j] < lb[j]:
                        trial[j] = lb[j]

            # Evaluate trial
            trial_fit = func(trial)
            evals += 1

            # Greedy selection
            if trial_fit <= fit[i]:
                pop[i] = trial
                fit[i] = trial_fit
                # Also inherit the winning parameters
                F[i] = F[i]  # unchanged (keep the updated value)
                CR[i] = CR[i]

            # Move to next individual, wrap around
            i = (i + 1) % NP

        # Find best solution
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]
        return best_x, best_y
