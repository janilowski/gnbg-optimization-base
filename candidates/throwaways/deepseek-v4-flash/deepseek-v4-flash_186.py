# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a self-adaptive Differential Evolution (jDE) for continuous black-box minimization. It adapts the mutation factor F and crossover rate CR per individual, uses rand/1/bin mutation, reflects out-of-bounds coordinates, and tracks the overall best solution.
# Search state: A population of candidate vectors (size pop_size) stored as a 2D array, plus arrays of per-individual F and CR values. The best-so-far solution and its objective value are maintained.
# Candidate generation: For each target vector, three distinct random population members (different from the target) are selected. A mutant vector is formed as a + F * (b - c). The trial vector is constructed via binomial crossover with the target using the individual's CR.
# Selection and replacement: The trial vector is evaluated. If its objective value is less than or equal to the target's, it replaces the target in the population; otherwise the target remains. The best solution is updated accordingly.
# Adaptation: Every generation, with probabilities tau1 and tau2, each individual's F and CR are updated by sampling from uniform distributions in [0.1, 1] and [0, 1] respectively.
# Exploration mechanisms: The mutation operator and the adaptive F (which can be large) promote exploration. Random selection of donor vectors ensures diversity.
# Exploitation mechanisms: The selection step preserves better solutions. The best-so-far solution is always retained, and low CR values encourage exploitation by preserving more target components.
# Boundary handling: After mutation and crossover, trial coordinates that exceed bounds are reflected symmetrically around the boundary into the feasible region.
# Budget strategy: The population size is set based on budget (max(5, budget//10) capped at 100). The number of generations is computed so that total evaluations (initial + per-generation) do not exceed the budget. For very small budgets, a pure random search fallback is used.
# Closest known influences: jDE (self-adaptive Differential Evolution) by Brest et al. (2006). The implementation is a stripped-down version with reflection boundary handling.
# Novelty or unusual aspects: None; it is a standard jDE variant chosen for its simplicity and robustness.
# Failure modes: May converge prematurely on highly multimodal landscapes with limited budget. Random search fallback is poor for high-dimensional problems with very small budgets.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lo = np.asarray(func.lower, dtype=float)
            hi = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lo = np.asarray(func.bounds.lb, dtype=float)
            hi = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide .lower/.upper or .bounds.lb/.bounds.ub")

        # Ensure bounds are 1-D arrays
        lo = lo.ravel()
        hi = hi.ravel()

        n = self.dim
        budget = self.budget

        # Minimal fallback for tiny budgets
        if budget < 5:
            best_x = np.random.uniform(lo, hi)
            best_y = func(best_x)
            for _ in range(budget - 1):
                x = np.random.uniform(lo, hi)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x
            return best_x, best_y

        # Population size – compromise between diversity and generations
        pop_size = max(5, min(100, budget // 10))

        # Number of generations: initial pop + pop_size * generations <= budget
        # We want at least one generation, but avoid overspending.
        max_gen = max(0, (budget // pop_size) - 1)
        # If we cannot even do one full generation, fall back to random search
        if max_gen == 0 and pop_size > budget:
            # budget is between 5 and pop_size-1, just sample budget points
            best_x = np.random.uniform(lo, hi)
            best_y = func(best_x)
            for _ in range(budget - 1):
                x = np.random.uniform(lo, hi)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x
            return best_x, best_y

        # Initialize population uniformly in bounds
        pop = np.random.uniform(lo, hi, size=(pop_size, n))

        # Evaluate initial population
        fits = np.empty(pop_size)
        for i in range(pop_size):
            fits[i] = func(pop[i])

        evals_done = pop_size
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_y = fits[best_idx]

        # jDE parameters
        tau1 = 0.1
        tau2 = 0.1
        F_low, F_high = 0.1, 1.0
        CR_low, CR_high = 0.0, 1.0

        # Initialize F and CR for each individual
        F = np.random.uniform(F_low, F_high, size=pop_size)
        CR = np.random.uniform(CR_low, CR_high, size=pop_size)

        generation = 0
        while evals_done < budget and generation < max_gen:
            generation += 1
            # For each target vector
            for i in range(pop_size):
                if evals_done >= budget:
                    break
                # Adapt F and CR with small probabilities
                if np.random.rand() < tau1:
                    F[i] = np.random.uniform(F_low, F_high)
                if np.random.rand() < tau2:
                    CR[i] = np.random.uniform(CR_low, CR_high)

                # Choose three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r = np.random.choice(candidates, size=3, replace=False)
                a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]

                # Mutation
                mutant = a + F[i] * (b - c)

                # Binomial crossover with target i
                cross_points = np.random.rand(n) < CR[i]
                if not cross_points.any():
                    cross_points[np.random.randint(0, n)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Boundary reflection (reflect back into [lo, hi])
                # For each dimension, reflect if out of bounds
                # Reflection: if x < lo, x = lo + (lo - x); if x > hi, x = hi - (x - hi)
                out_low = trial < lo
                trial[out_low] = lo[out_low] + (lo[out_low] - trial[out_low])
                out_high = trial > hi
                trial[out_high] = hi[out_high] - (trial[out_high] - hi[out_high])
                # In rare case of overflow, clip
                np.clip(trial, lo, hi, out=trial)

                # Evaluate trial
                trial_val = func(trial)
                evals_done += 1

                # Selection
                if trial_val <= fits[i]:
                    pop[i] = trial
                    fits[i] = trial_val
                    if trial_val < best_y:
                        best_y = trial_val
                        best_x = trial.copy()

        return best_x, best_y
