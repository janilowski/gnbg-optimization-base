# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a self‑adaptive Differential Evolution (jDE) algorithm for
# black‑box minimization. The algorithm maintains a population of candidate
# solutions, each with its own mutation factor (F) and crossover rate (CR).
# These parameters are evolved along with the solutions.
#
# Search state: A population of NP vectors, each with associated fitness value,
# F, and CR. The current best solution is stored separately.
#
# Candidate generation: For each target vector xi, a mutant vector is created
# using DE/rand/1: v = x_r1 + F_i * (x_r2 - x_r3), where r1, r2, r3 are distinct
# random indices different from i. A trial vector is formed via binomial
# crossover with rate CR_i.
#
# Selection and replacement: Greedy selection – if the trial vector has lower
# (or equal) fitness than the target, it replaces the target. The F and CR that
# produced the successful trial are inherited; otherwise the old F and CR are
# retained.
#
# Adaptation: F and CR are self‑adapted per individual. With probabilities
# tau1=0.1 and tau2=0.1, new F and CR values are sampled from uniform
# distributions (F in [0.1, 1], CR in [0, 1]) each generation. These new values
# are used for the current trial; if the trial succeeds, they replace the old
# ones, else they are discarded.
#
# Exploration mechanisms: Mutation with large F and high CR encourages
# exploration. Newly sampled F and CR values can increase diversity. The
# population maintains a range of behaviours.
#
# Exploitation mechanisms: As F and CR adapt via successful trials, they
# converge to values that yield good solutions. Greedy selection pushes the
# population toward better regions.
#
# Boundary handling: Trial vectors are clipped component‑wise to the domain
# bounds.
#
# Budget strategy: The budget is allocated into an initial population evaluation
# (NP evaluations) and then generation‑by‑generation evaluations until
# exhaustion. Population size is chosen as NP = max(2, min(10*dim, budget//10))
# to allow a reasonable number of generations.
#
# Closest known influences: jDE (Brest et al., 2006), a variant of Differential
# Evolution with self‑adaptive control parameters.
#
# Novelty or unusual aspects: No novelty; this is a standard implementation of
# jDE.
#
# Failure modes: May perform poorly on highly multimodal or deceptive landscapes
# if population size is too small or budget insufficient. Clipping can cause
# stagnation if the optimum lies on the boundary. Overhead of per‑individual
# parameter adaptation adds minimal cost.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    """Self‑adaptive Differential Evolution (jDE) for black‑box minimization."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # Use numpy's global random state; the harness sets the seed before
        # each call.

    def __call__(self, func):
        # ----- read bounds -----
        try:
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        if lower.shape == ():
            lower = np.full(self.dim, lower)
            upper = np.full(self.dim, upper)
        lb = lower
        ub = upper

        # ----- handle tiny budgets -----
        budget_left = self.budget
        if budget_left < 2:
            # only one evaluation possible
            x = lb + (ub - lb) * np.random.random(self.dim)
            y = func(x)
            return x, y

        # ----- population size -----
        np_pop = max(2, min(10 * self.dim, budget_left // 10))
        np_pop = min(np_pop, budget_left)

        # ----- initialisation -----
        pop = lb + (ub - lb) * np.random.uniform(0, 1, (np_pop, self.dim))
        fitness = np.array([func(p) for p in pop])
        budget_left -= np_pop

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # ----- control parameter initialisation (jDE style) -----
        F = np.random.uniform(0.1, 1.0, np_pop)
        CR = np.random.uniform(0, 1, np_pop)
        tau1 = 0.1
        tau2 = 0.1

        # ----- main loop -----
        while budget_left > 0:
            for i in range(np_pop):
                # generate new F_i, CR_i with probabilities tau1, tau2
                if np.random.random() < tau1:
                    F_i = np.random.uniform(0.1, 1.0)
                else:
                    F_i = F[i]
                if np.random.random() < tau2:
                    CR_i = np.random.uniform(0, 1)
                else:
                    CR_i = CR[i]

                # select three distinct random indices different from i
                candidates = [j for j in range(np_pop) if j != i]
                np.random.shuffle(candidates)
                r1, r2, r3 = candidates[:3]

                # mutation (DE/rand/1)
                mutant = pop[r1] + F_i * (pop[r2] - pop[r3])

                # binomial crossover
                trial = pop[i].copy()
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.random() < CR_i or j == jrand:
                        trial[j] = mutant[j]

                # boundary handling: clip to domain
                trial = np.clip(trial, lb, ub)

                # evaluation
                trial_f = func(trial)
                budget_left -= 1

                # selection
                if trial_f <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    F[i] = F_i
                    CR[i] = CR_i
                    if trial_f < best_y:
                        best_y = trial_f
                        best_x = trial.copy()

                # stop if budget exhausted
                if budget_left == 0:
                    break

        return best_x, best_y
