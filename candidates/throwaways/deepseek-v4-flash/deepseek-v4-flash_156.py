# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (rand/1/bin) with bounded reflection. Designed for black‑box minimization.
# Search state: A fixed‑size population of candidate solutions (real vectors) evolving over generations.
# Candidate generation: For each population member (target), three distinct random individuals are chosen.
#   The mutant vector is target + F * (individual1 - individual2 + individual3 - individual4)
#   with F = 0.8. Crossover (binomial, CR = 0.9) mixes the mutant with the target.
# Selection and replacement: Greedy selection – if the trial vector has lower objective value than the target,
#   it replaces the target in the next population.
# Adaptation: None – parameters F and CR are fixed.
# Exploration mechanisms: Differential mutation with random differentials and binomial crossover create
#   diverse trial points.
# Exploitation mechanisms: Elitist replacement (stochastic, not fully greedy) and the use of the population
#   mean as a base for mutation drives convergence.
# Boundary handling: Components violating [lower, upper] are reflected back into the feasible domain.
# Budget strategy: Population size is set to min(5*dim, 20) but at least 4. Number of generations is
#   floor((budget - pop_size) / pop_size). The algorithm stops exactly when the budget is exhausted.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997) with rand/1/bin.
# Novelty or unusual aspects: Reflection boundary handling instead of clamping; population size scaled
#   sublinearly with dimension to keep within budget.
# Failure modes: May stagnate on highly multimodal landscapes if F or CR are poorly suited; population
#   size too small for high dimensions can lose diversity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Differential Evolution (rand/1/bin) minimizer for black‑box functions."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ---- read bounds ----------------------------------------------------
        try:
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lower = np.asarray(func.bounds.lb, dtype=float)
                upper = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise TypeError("Cannot determine bounds from func")
        if lower.ndim == 0:
            lower = lower * np.ones(self.dim)
            upper = upper * np.ones(self.dim)

        # ---- population setup ------------------------------------------------
        pop_size = max(4, min(5 * self.dim, 20))
        pop_size = min(pop_size, self.budget)          # ensure we never need more evals than budget
        max_generations = (self.budget - pop_size) // pop_size  # floor division

        # initial population uniformly in [lower, upper]
        pop = lower + (upper - lower) * np.random.rand(pop_size, self.dim)
        fitness = np.empty(pop_size)
        for i in range(pop_size):
            fitness[i] = func(pop[i])
        evals = pop_size

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # ---- DE parameters --------------------------------------------------
        F = 0.8
        CR = 0.9

        # ---- main loop -------------------------------------------------------
        for _ in range(max_generations):
            if evals >= self.budget:
                break
            # for each target vector
            for i in range(pop_size):
                # select three distinct random indices != i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                # mutant = pop[r1] + F * (pop[r2] - pop[r3])   (rand/1)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # binomial crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[i])
                # ensure at least one component from mutant
                trial[j_rand] = mutant[j_rand]
                # boundary reflection
                trial = np.where(trial < lower, 2 * lower - trial, trial)
                trial = np.where(trial > upper, 2 * upper - trial, trial)
                # re‑project just in case reflection overshoots
                trial = np.clip(trial, lower, upper)
                # evaluate
                trial_fit = func(trial)
                evals += 1
                # selection
                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()
                if evals >= self.budget:
                    break

        return best_x, best_y
