import numpy as np
from typing import Tuple, Any

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a Differential Evolution (DE) optimizer
#          with a rand/1/bin strategy for black-box minimization.
# Search state: A population of candidate solutions stored as a matrix
#               (pop_size x dim), plus the corresponding fitness vector.
# Candidate generation: For each target vector, a mutant is created by
#                       adding the scaled difference of two random
#                       population members to a third.  A binomial
#                       crossover between the target and mutant yields
#                       the trial vector.
# Selection and replacement: Greedy selection: the trial vector replaces
#                            the target if its fitness is better (lower).
# Adaptation: The step-size parameter F and crossover rate CR are fixed
#             (F=0.5, CR=0.9).  No adaptive tuning is performed.
# Exploration mechanisms: The fixed, moderate F and CR together with the
#                         random differential mutation provide continuous
#                         exploration across the search space.
# Exploitation mechanisms: As the population converges, difference vectors
#                          shrink, leading to finer local search.  The
#                          elitist best-so-far is stored separately.
# Boundary handling: Trial coordinates outside the domain are reflected
#                    back into the allowed range.
# Budget strategy: The total number of function evaluations is strictly
#                  limited to the given budget.  The loop exits as soon
#                  as the budget is exhausted, even if a generation is
#                  incomplete.
# Closest known influences: Standard differential evolution (Storn & Price,
#                           1997) with binomial crossover and fixed
#                           control parameters.
# Novelty or unusual aspects: None; a straightforward classical DE
#                             implementation.
# Failure modes: On strongly multimodal or deceptive landscapes, DE with
#                fixed parameters may stagnate or converge prematurely.
#                Very high-dimensional problems with limited budget may
#                prevent adequate exploration.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Differential Evolution (DE) minimizer for black-box functions."""

    def __init__(self, budget: int, dim: int) -> None:
        """
        Prepare the optimizer.

        Args:
            budget: Maximum number of function evaluations.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # DE control parameters (fixed)
        self.F = 0.5          # scaling factor
        self.CR = 0.9         # crossover probability

        # Population size: at least 4 (required for mutation) and
        # scaled by dim, but clamped to avoid exceeding budget too quickly.
        # A common rule is pop_size = 10 * dim, but we also ensure we
        # can run at least a few generations.
        self.pop_size = max(4, min(10 * dim, self.budget // 2))

        # Store bounds once __call__ is invoked
        self.lb = None
        self.ub = None

    def _get_bounds(self, func: Any) -> None:
        """Retrieve lower and upper bounds from the function object."""
        # Prefer func.lower / func.upper; fallback to func.bounds.lb / .ub
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            self.lb = np.asarray(func.lower, dtype=float)
            self.ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            if hasattr(b, 'lb') and hasattr(b, 'ub'):
                self.lb = np.asarray(b.lb, dtype=float)
                self.ub = np.asarray(b.ub, dtype=float)
            else:
                raise AttributeError("Cannot determine bounds from func")
        else:
            raise AttributeError("Function object has no 'lower'/'upper' or 'bounds.lb'/'bounds.ub'")

        # Ensure they are 1-D arrays
        self.lb = self.lb.ravel()
        self.ub = self.ub.ravel()
        if self.lb.shape[0] != self.dim or self.ub.shape[0] != self.dim:
            raise ValueError("Bound dimensions do not match dim")

    def __call__(self, func: Any) -> Tuple[np.ndarray, float]:
        """
        Run the optimizer on the given function.

        Args:
            func: A callable object with methods .lower/.upper or .bounds.
                  It must accept a 1-D array and return a scalar (minimization).

        Returns:
            (best_x, best_y) where best_x is the best found solution (1-D array)
            and best_y is its objective value.
        """
        self._get_bounds(func)

        # Initialize population uniformly in [lb, ub]
        pop = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))

        # Evaluate initial population
        fitness = np.full(self.pop_size, np.inf)
        evaluations = 0
        best_x = None
        best_y = np.inf

        for i in range(self.pop_size):
            if evaluations >= self.budget:
                break
            y = func(pop[i])
            fitness[i] = y
            evaluations += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Main DE loop
        while evaluations < self.budget:
            # For each target vector, generate a trial
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                # Choose three distinct random indices != i
                candidates = [j for j in range(self.pop_size) if j != i]
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutation: DE/rand/1
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])

                # Binomial crossover
                j_rand = np.random.randint(0, self.dim)
                trial = np.array([mutant[j] if (np.random.random() < self.CR or j == j_rand)
                                  else pop[i][j] for j in range(self.dim)])

                # Boundary handling: reflect trial back into domain
                # Using simple reflection: if out of bounds, reflect off the boundary.
                # This avoids clamping and helps maintain diversity.
                low = self.lb
                high = self.ub
                trial = np.where(trial < low, 2 * low - trial, trial)
                trial = np.where(trial > high, 2 * high - trial, trial)
                # In rare cases reflection may still be outside; then clamp
                trial = np.clip(trial, self.lb, self.ub)

                # Evaluate trial (if budget remains)
                y_trial = func(trial)
                evaluations += 1

                # Greedy selection
                if y_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = y_trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()

        return best_x, best_y
