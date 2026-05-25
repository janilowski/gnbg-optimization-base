import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a Differential Evolution (DE) algorithm with rand/1/bin strategy for black-box minimization.
# Search state: Maintains a population of candidate solutions (NP individuals) and their fitness values.
# Candidate generation: For each target individual, a mutant vector is created by adding the scaled difference of two random individuals to a third. Binomial crossover produces a trial vector.
# Selection and replacement: Greedy selection: trial replaces target if it yields lower (better) objective value.
# Adaptation: Fixed mutation factor F=0.9 and crossover rate CR=0.9. Population size NP is set based on budget to allow multiple generations.
# Exploration mechanisms: Mutation with random differentials and crossover promote diversity.
# Exploitation mechanisms: Greedy selection preserves best solutions and differentials exploit population geometry.
# Boundary handling: Trial vectors are clipped to the bounds.
# Budget strategy: Number of evaluations is tracked explicitly; generations stop when budget exhausted. Initial population and each trial evaluation count.
# Closest known influences: Classic DE (Storn & Price, 1997).
# Novelty or unusual aspects: Simple parameter choices and budget‑aware population sizing.
# Failure modes: May converge prematurely on rugged landscapes due to fixed parameters. Small budget may limit exploration.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    """Differential Evolution (DE) minimizer for black‑box functions."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ----- read bounds -----
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        d = self.dim
        budget = self.budget

        # ----- decide population size -----
        # aim for at least 10 generations, but not too large
        NP = max(4, int(budget / 20))
        if NP > budget:
            NP = budget  # at most one evaluation per individual
        # fallback for very tiny budgets
        if NP < 4 or budget < 4:
            best_x = np.random.uniform(lb, ub, size=d)
            best_y = func(best_x)
            evals = 1
            while evals < budget:
                x = np.random.uniform(lb, ub, size=d)
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # ----- initialisation -----
        pop = np.random.uniform(lb, ub, size=(NP, d))
        fitness = np.empty(NP)
        for i in range(NP):
            fitness[i] = func(pop[i])
        evals = NP

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # ----- DE parameters -----
        F = 0.9
        CR = 0.9

        # number of full generations we can afford
        remaining_evals = budget - evals
        max_gen = remaining_evals // NP
        if max_gen <= 0:
            max_gen = 1  # still attempt one generation, may break when budget runs out

        # ----- main DE loop -----
        for gen in range(max_gen):
            for i in range(NP):
                if evals >= budget:
                    break

                # select three distinct indices != i
                indices = list(range(NP))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # mutation
                mutant = pop[a] + F * (pop[b] - pop[c])

                # binomial crossover
                jrand = np.random.randint(d)
                mask = np.random.rand(d) < CR
                trial = np.where(mask, mutant, pop[i])
                trial[jrand] = mutant[jrand]  # ensure at least one dimension from mutant

                # bound handling: clip
                trial = np.clip(trial, lb, ub)

                # evaluation
                trial_fit = func(trial)
                evals += 1

                # selection
                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

            if evals >= budget:
                break

        return best_x, best_y
