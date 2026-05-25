# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) optimizer for black-box minimization.
#          It follows the DE/rand/1/bin scheme, which is robust and parameter-light.
# Search state: A population of candidate solutions (x vectors) and their objective values.
# Candidate generation: For each target individual, a mutant is created by adding the
#     scaled difference of two random distinct population members to a third.
#     Then, binomial crossover combines the mutant with the target to produce a trial vector.
# Selection and replacement: The trial vector replaces the target if its objective value
#     is lower (minimization). Replacement is immediate (greedy).
# Adaptation: Fixed parameters (F=0.7, CR=0.9); no online adaptation to keep code simple.
# Exploration mechanisms: The mutation operator uses scaled vector differences, which
#     naturally drives exploration across the search space.
# Exploitation mechanisms: The selection and replacement step gradually improves the
#     population; the population convergence over generations provides local refinement.
# Boundary handling: Trial vectors are clipped component‑wise to the search domain.
# Budget strategy: The population size is chosen adaptively as a function of the dimension
#     and the budget, ensuring at least 2 generations are performed.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None; a straightforward implementation with minimal tuning.
# Failure modes: May stagnate on highly multimodal or deceptive functions when budget is
#     very small; performance depends on the population size scaling heuristic.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

        # Algorithm parameters (fixed throughout the run)
        self.F = 0.7          # mutation factor
        self.CR = 0.9         # crossover probability

        # Adaptive population size: aim for at least 2 generations.
        # Ensure at least 10 individuals, at most floor(budget/2).
        min_pop = max(10, 2 * dim)
        self.popsize = min(min_pop, self.budget // 2)

    def __call__(self, func):
        # ---------- Retrieve bounds ----------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via .lower/.upper or .bounds.lb/.bounds.ub")

        # Ensure bounds are 1-D arrays of correct dimension
        lower = lower.ravel() if lower.ndim > 1 else lower
        upper = upper.ravel() if upper.ndim > 1 else upper
        if lower.shape[0] != self.dim or upper.shape[0] != self.dim:
            raise ValueError("Bound arrays must have length equal to dim")

        # ---------- Initialization ----------
        pop = np.random.uniform(lower, upper, size=(self.popsize, self.dim))
        fit = np.full(self.popsize, np.inf)
        evals_used = 0

        # Evaluate initial population
        for i in range(self.popsize):
            if evals_used >= self.budget:
                break
            fit[i] = func(pop[i])
            evals_used += 1

        # Track best overall
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # ---------- Main DE loop ----------
        generation = 0
        while evals_used < self.budget:
            generation += 1
            # For each target individual generate a trial vector
            for i in range(self.popsize):
                if evals_used >= self.budget:
                    break

                # Step 1: Mutation (DE/rand/1)
                # Pick three distinct random indices different from i
                candidates = list(range(self.popsize))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])

                # Step 2: Crossover (binomial)
                # Always mutate at least one component (random index)
                trial = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # Step 3: Boundary handling (clip to bounds)
                trial = np.clip(trial, lower, upper)

                # Step 4: Evaluate trial
                trial_fit = func(trial)
                evals_used += 1

                # Step 5: Selection (greedy)
                if trial_fit < fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fit
                    # Update global best if improved
                    if trial_fit < best_y:
                        best_x = trial.copy()
                        best_y = trial_fit

        return best_x, best_y
