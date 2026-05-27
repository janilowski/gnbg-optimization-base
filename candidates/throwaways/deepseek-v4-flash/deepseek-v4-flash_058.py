import numpy as np
from collections.abc import Callable
from typing import Tuple, Optional

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE/rand/1/bin) variant for black-box minimization.
#          It combines global exploration through difference vectors with local refinement via
#          binomial crossover and greedy selection.
# Search state: A population of candidate vectors and their corresponding fitness values.
# Candidate generation: For each target vector, three distinct individuals are selected uniformly
#                       from the population. A mutant is created as base + F * (diff1 - diff2).
#                       F is randomly sampled per target in [0.5, 1.0) (dithering) to improve
#                       diversity. A trial vector is produced via binomial crossover with the target.
# Selection and replacement: The trial replaces the target if it yields a lower (better) fitness.
# Adaptation: The mutation scaling factor F is jittered per individual (no other adaptation).
# Exploration mechanisms: Random selection of base and difference vectors, dithering of F,
#                         and crossover mixing promote wide exploration of the search space.
# Exploitation mechanisms: Greedy replacement preserves good solutions; the population gradually
#                          concentrates on promising regions.
# Boundary handling: Trial vectors are clipped component-wise to the bounds [lb, ub] after mutation.
# Budget strategy: The initial population uses the first min(budget, popsize) evaluations.
#                  Subsequent evaluations are used one by one (one trial per evaluation) in a
#                  round‑robin order until the budget is exhausted. This guarantees exactly
#                  budget function calls.
# Closest known influences: Classic DE/rand/1/bin algorithm (Storn & Price, 1997) with jitter.
# Novelty or unusual aspects: None; a straightforward, robust implementation.
# Failure modes: For very small budgets (< 4 evaluations) it falls back to pure random search.
#                On highly multimodal or deceptive landscapes the algorithm may converge prematurely
#                if the population loses diversity. High dimensions may require a larger population
#                size which is limited by the budget.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Differential Evolution minimizer for black‑box functions."""

    def __init__(self, budget: int, dim: int):
        """
        Args:
            budget: Maximum number of allowed function evaluations.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func: Callable) -> Tuple[np.ndarray, float]:
        # ------------------------------------------------------------------
        # 1. Read bounds
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.atleast_1d(np.asarray(func.lower, dtype=float))
            ub = np.atleast_1d(np.asarray(func.upper, dtype=float))
        elif hasattr(func, 'bounds'):
            b = func.bounds
            if hasattr(b, 'lb') and hasattr(b, 'ub'):
                lb = np.atleast_1d(np.asarray(b.lb, dtype=float))
                ub = np.atleast_1d(np.asarray(b.ub, dtype=float))
            else:
                raise ValueError("Cannot find lower/upper bounds from func")
        else:
            raise ValueError("func must provide .lower/.upper or .bounds.lb/.bounds.ub")

        dimension = self.dim
        budget = self.budget

        # ------------------------------------------------------------------
        # 2. Set population size
        # ------------------------------------------------------------------
        # For very small budgets we do random sampling; otherwise use a
        # population that scales with dimension but is limited by the budget.
        if budget < 4:
            # Pure random search with available budget
            best_x = np.empty(dimension)
            best_y = np.inf
            for _ in range(budget):
                x = np.random.uniform(lb, ub, size=dimension)
                y = float(func(x))
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # reasonable population size
        popsize = max(4, min(budget // 2, 4 * dimension))
        # Ensure we do not exceed budget for initial evaluation
        if popsize > budget:
            popsize = budget

        # ------------------------------------------------------------------
        # 3. Initialise population
        # ------------------------------------------------------------------
        pop = np.random.uniform(lb, ub, size=(popsize, dimension))
        fits = np.empty(popsize)
        for i in range(popsize):
            fits[i] = float(func(pop[i]))

        evals_used = popsize
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_y = fits[best_idx]

        # ------------------------------------------------------------------
        # 4. Main DE loop (one evaluation per trial)
        # ------------------------------------------------------------------
        # DE parameters
        CR = 0.9        # crossover probability
        F_low = 0.5     # lower bound for jittered F
        F_high = 1.0    # upper bound for jittered F

        index = 0                     # round‑robin target index
        while evals_used < budget:
            # Select three distinct individuals different from the target
            candidates = list(range(popsize))
            candidates.remove(index)
            np.random.shuffle(candidates)
            a, b, c = candidates[:3]

            # Mutation with jittered F
            F = np.random.uniform(F_low, F_high)
            mutant = pop[a] + F * (pop[b] - pop[c])
            # Boundary handling: clip to [lb, ub]
            mutant = np.clip(mutant, lb, ub)

            # Binomial crossover
            trial = pop[index].copy()
            j_rand = np.random.randint(dimension)
            for j in range(dimension):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # Evaluate trial
            trial_y = float(func(trial))
            evals_used += 1

            # Selection: replace if better
            if trial_y < fits[index]:
                pop[index] = trial
                fits[index] = trial_y
                if trial_y < best_y:
                    best_y = trial_y
                    best_x = trial.copy()

            # Move to next target (round‑robin)
            index = (index + 1) % popsize

        return best_x, best_y
