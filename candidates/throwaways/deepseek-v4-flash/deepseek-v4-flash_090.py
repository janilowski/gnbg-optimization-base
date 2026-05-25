# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A differential evolution (DE/rand/1/bin) minimizer with fixed population size and budget control.
# Search state: A population of candidate solutions (NP vectors) and their fitness values, plus the global best.
# Candidate generation: For each target vector, a mutant is created by adding the scaled difference of two random distinct population members to a third distinct member. Binomial crossover combines the mutant with the target.
# Selection and replacement: Greedy – trial replaces target if trial has lower (better) objective value.
# Adaptation: No automatic parameter adaptation; F and CR are fixed (0.8 and 0.9).
# Exploration mechanisms: Mutation via random differentials provides diversity; crossover mixes components.
# Exploitation mechanisms: Population gradually converges as better solutions replace worse ones; global best is stored.
# Boundary handling: Mutated coordinates outside [lower, upper] are reflected back into the domain using mirroring.
# Budget strategy: Population size is capped so that at least one full generation can be performed. Evaluations are counted strictly; the algorithm stops as soon as the budget is exhausted, even mid-generation.
# Closest known influences: Classic DE/rand/1/bin (Storn & Price, 1997).
# Novelty or unusual aspects: Minimal – only a straightforward DE implementation with budget compliance and reflection boundary handling.
# Failure modes: High-dimensional problems may require larger populations than budget allows; convergence can stall if F and CR are unsuitable for the landscape. Unimodal, separable functions are handled efficiently; highly multimodal or deceptive landscapes may cause premature convergence.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """A black-box minimizer using differential evolution (DE/rand/1/bin)."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Read bounds from the provided function object
        # Prefer func.lower/upper; fallback to func.bounds.lb/ub
        try:
            lb = getattr(func, 'lower', None)
            ub = getattr(func, 'upper', None)
            if lb is None or ub is None:
                bounds = getattr(func, 'bounds', None)
                if bounds is not None:
                    lb = bounds.lb
                    ub = bounds.ub
                else:
                    raise AttributeError
        except AttributeError:
            raise ValueError("Cannot read bounds from func object")

        lb = np.asarray(lb, dtype=float).reshape(1, -1)
        ub = np.asarray(ub, dtype=float).reshape(1, -1)

        # Determine population size NP
        # We need NP to be at least 4 for DE mutation, and we want at least one full generation.
        # Budget must cover NP initial evaluations + up to NP * gen evaluations.
        # We set NP as large as possible subject to: budget >= NP + NP*k (k>=1) -> NP <= budget/2
        # but also capped at 30 for stability and limited to dim*10.
        possible_np = min(30, max(4, min(self.dim * 10, self.budget // 2)))
        # Ensure NP is even? Not necessary.
        NP = possible_np
        if NP < 4:
            NP = self.budget  # fallback: just random sampling if budget tiny

        # DE parameters
        F = 0.8
        CR = 0.9

        # Initialise population
        pop = np.random.uniform(lb, ub, size=(NP, self.dim))
        fitness = np.full(NP, np.inf)

        evals = 0
        best_x = None
        best_y = np.inf

        # Evaluate initial population
        for i in range(NP):
            if evals >= self.budget:
                # Not enough budget to evaluate all; trim population
                pop = pop[:evals]
                fitness = fitness[:evals]
                NP = evals
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()

        # Early exit if no evaluations left
        if evals >= self.budget:
            return best_x, best_y

        # Main DE loop
        generation = 0
        while evals < self.budget:
            # For each target vector, generate a trial vector
            for i in range(NP):
                if evals >= self.budget:
                    break

                # Choose three distinct random indices different from i
                r = np.random.choice(NP, 3, replace=False)
                a, b, c = r
                # Ensure they are distinct from i (already guaranteed by choice)
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                trial = np.where(np.random.rand(self.dim) < CR,
                                 mutant,
                                 pop[i])

                # Boundary handling: reflect
                trial = np.where(trial < lb, 2 * lb - trial, trial)
                trial = np.where(trial > ub, 2 * ub - trial, trial)
                # Clip to bounds in case reflection goes wild (rare)
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_f = func(trial)
                evals += 1

                # Selection: keep better
                if trial_f < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_y:
                        best_y = trial_f
                        best_x = trial.copy()

            generation += 1

        return best_x, best_y
