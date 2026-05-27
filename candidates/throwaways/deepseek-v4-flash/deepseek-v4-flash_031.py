import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a Differential Evolution (DE/rand/1/bin) algorithm
#          for black-box minimization. It is designed to be compact, robust across
#          dimensions, and to strictly respect the evaluation budget.
# Search state: A population of candidate solutions (real vectors) stored in a list,
#               together with their objective values. The global best solution and its
#               value are tracked separately.
# Candidate generation: For each target vector, a mutant is created by adding the
#                       scaled difference of two randomly chosen distinct population
#                       members to a third distinct member (rand/1). The scale factor
#                       F is sampled uniformly from [0.5, 1.0] per individual (dither).
#                       Then binomial crossover with probability CR = 0.9 is applied
#                       to combine the mutant with the target, ensuring at least one
#                       component from the mutant.
# Selection and replacement: Deterministic replacement – the trial vector replaces the
#                            target if and only if its objective value is lower (better).
# Adaptation: No parameter adaptation; F and CR are fixed (F is randomly varied per
#              individual but within a fixed range, which provides a form of adaptation).
# Exploration mechanisms: The mutation operator with dither maintains diversity;
#                         crossover can produce solutions far from parents.
# Exploitation mechanisms: The greedy selection preserves the best solutions;
#                          the population converges over generations.
# Boundary handling: Components that exceed the variable bounds are clipped to the
#                    nearest bound.
# Budget strategy: The population size NP is chosen as max(4, min(10*dim, budget//4))
#                  to ensure enough evaluations for at least one generation. The
#                  algorithm runs generation by generation until the budget is exhausted.
#                  If budget is too small for a full initial population, all budget
#                  evaluations are used to sample random points.
# Closest known influences: Differential Evolution (Storn & Price, 1997),
#                           dither variant (random F per individual).
# Novelty or unusual aspects: None – straightforward DE implementation.
# Failure modes: May converge prematurely on highly multimodal landscapes,
#                especially with small population size or budget. High-dimensional
#                problems may require a larger population than budget allows.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Black-box minimization using Differential Evolution (DE/rand/1/bin)."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # Determine population size: at least 4, at most 10*dim, and respecting budget.
        self.NP = max(4, min(10 * dim, budget // 4))
        # If budget is extremely small, evaluate as many random points as possible.
        if self.NP > budget:
            self.NP = budget
        self.F_low = 0.5   # lower bound for dither
        self.F_high = 1.0  # upper bound
        self.CR = 0.9

    def __call__(self, func):
        # Read bounds from the function object.
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lower = np.asarray(b.lb, dtype=float)
            upper = np.asarray(b.ub, dtype=float)
        else:
            raise ValueError("Cannot read bounds from func object.")

        # Flatten to 1-D array of length self.dim.
        lower = np.broadcast_to(lower, (self.dim,))
        upper = np.broadcast_to(upper, (self.dim,))

        # Initialize evaluation counter and best-so-far.
        evals = 0
        best_x = None
        best_y = np.inf

        # Helper to evaluate a single point, update best and evals.
        def evaluate(x):
            nonlocal evals, best_x, best_y
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.copy(x)
            return y

        # --- Initial population ---
        pop = []
        fit = []
        for _ in range(self.NP):
            if evals >= self.budget:
                break
            x = np.random.uniform(lower, upper)
            y = evaluate(x)
            pop.append(x)
            fit.append(y)

        # Adjust NP to the actual number of initial individuals evaluated.
        NP = len(pop)

        # If budget exhausted after initial population, return best.
        if evals >= self.budget:
            return best_x, best_y

        # --- Main generation loop ---
        while evals < self.budget:
            # Keep a copy of the current population for mutation (no in-place contamination).
            old_pop = [np.copy(p) for p in pop]
            old_fit = fit[:]  # actually not used for selection, only for reference

            # Process each target individual in order.
            for i in range(NP):
                if evals >= self.budget:
                    break

                # Choose three distinct random indices different from i.
                candidates = list(range(NP))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutation: donor = old_pop[r1] + F * (old_pop[r2] - old_pop[r3])
                F = np.random.uniform(self.F_low, self.F_high)
                donor = old_pop[r1] + F * (old_pop[r2] - old_pop[r3])

                # Binomial crossover with at least one component from donor.
                trial = np.copy(old_pop[i])
                j_rand = np.random.randint(0, self.dim)
                for j in range(self.dim):
                    if np.random.random() < self.CR or j == j_rand:
                        trial[j] = donor[j]

                # Clip to bounds.
                trial = np.clip(trial, lower, upper)

                # Evaluate trial.
                trial_y = evaluate(trial)

                # Selection: replace if better.
                if trial_y < old_fit[i]:
                    pop[i] = trial
                    fit[i] = trial_y

        return best_x, best_y
