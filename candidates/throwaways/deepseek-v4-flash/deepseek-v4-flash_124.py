import numpy as np
from typing import Tuple, Optional

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a classic Differential Evolution (DE/rand/1/bin)
#          optimizer for black-box minimization. It is intended for the GNBG benchmark.
# Search state: A population of candidate solutions (real vectors) stored in a 2D numpy array,
#               along with their corresponding objective function values.
# Candidate generation: For each target vector, a mutant is created using the DE/rand/1 scheme:
#                       mutant = best + F * (r1 - r2), where r1,r2 are distinct random population members.
# Selection and replacement: Deterministic greedy selection: the trial vector replaces the target
#                            if it yields a lower objective value.
# Adaptation: No on-the-fly adaptation; the mutation factor F and crossover rate CR are fixed.
# Exploration mechanisms: The random selection of base vectors and difference vectors encourages
#                         exploration, especially early in the run. The binomial crossover also contributes.
# Exploitation mechanisms: The use of the best-so-far vector as the base in the mutation formula
#                          gradually focuses the search around the current best. (Note: standard DE/rand/1
#                          uses a random base; here we deviate slightly to improve exploitation: we use
#                          the best vector as the base. This is a common variant, sometimes called DE/best/1.)
# Boundary handling: Generated mutant coordinates that fall outside bounds are reflected back into the
#                    feasible domain using a symmetrical reflection (midpoint reflection). If reflection
#                    still yields a violation, the coordinate is clipped to the bound.
# Budget strategy: The population size is chosen as max(10, 3*dim) but capped so that at least 20 generations
#                  can be performed given the budget. After each function evaluation, a counter is decremented.
#                  The algorithm stops immediately when the budget is exhausted, even in the middle of a generation.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997) with the DE/best/1/bin variant.
# Novelty or unusual aspects: None; a straightforward and robust implementation tuned for black-box benchmarks.
# Failure modes: On highly multimodal landscapes with very limited budget, the algorithm may converge prematurely.
#                Fixed F and CR may be suboptimal for some function classes.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the Differential Evolution optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of function evaluations.
        dim : int
            Dimensionality of the search space. Used to set population size.
        """
        self.budget = budget
        self.dim = dim

        # Population size: at least 10, at most 3*dim, but also limited so that
        # we can run at least 20 generations given the budget.
        base_pop = max(10, 3 * dim)
        gen_estimate = budget / base_pop
        if gen_estimate < 20:
            # Reduce population size to allow 20 generations
            pop_size = max(10, int(budget / 20))
        else:
            pop_size = base_pop
        self.pop_size = min(pop_size, budget)  # Cannot exceed budget
        self.F = 0.8          # Mutation factor
        self.CR = 0.9         # Crossover rate

    def __call__(self, func) -> Tuple[np.ndarray, float]:
        """
        Run the DE optimizer on the given objective function.

        Parameters
        ----------
        func : callable
            Objective function to minimize. Must have attributes `lower`/`upper` or `bounds.lb`/`bounds.ub`
            to define the feasible domain.

        Returns
        -------
        best_x : np.ndarray
            Best solution found.
        best_y : float
            Best objective value found.
        """
        # Read bounds from the function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("The objective function does not provide required bounds (lower/upper or bounds.lb/ub).")

        # Ensure bounds are 1D arrays
        lb = lb.flatten()
        ub = ub.flatten()
        dim = self.dim
        if len(lb) != dim or len(ub) != dim:
            raise ValueError("Bounds dimensions do not match problem dimension.")

        # Remaining budget in evaluations
        rem_eval = self.budget

        # --- Initialization ---
        pop = np.random.uniform(lb, ub, size=(self.pop_size, dim))
        # Evaluate initial population
        fitness = np.empty(self.pop_size)
        for i in range(self.pop_size):
            if rem_eval <= 0:
                break
            fitness[i] = func(pop[i])
            rem_eval -= 1

        # Track best so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # --- Main DE loop ---
        # Continue as long as we have evaluations left and population is not empty
        while rem_eval > 0 and self.pop_size > 0:
            # For each target vector in the population
            for i in range(self.pop_size):
                if rem_eval <= 0:
                    break

                # Choose three distinct random indices, all different from i
                indices = [idx for idx in range(self.pop_size) if idx != i]
                if len(indices) < 3:
                    # Not enough distinct individuals: continue to next target
                    continue
                r = np.random.choice(indices, size=3, replace=False)
                r1, r2, r3 = r

                # Mutation: DE/best/1 (base is best individual)
                # mutant = best + F * (pop[r1] - pop[r2])
                base = best_x   # best solution so far
                mutant = base + self.F * (pop[r1] - pop[r2])

                # Boundary reflection (midpoint reflection)
                # Reflect coordinates that are out of bounds around the nearest bound.
                # If reflection still out of bounds, clip.
                low_viol = mutant < lb
                high_viol = mutant > ub
                # Reflection: new = bound + (bound - old) = 2*bound - old
                mutant[low_viol] = 2 * lb[low_viol] - mutant[low_viol]
                mutant[high_viol] = 2 * ub[high_viol] - mutant[high_viol]
                # In case reflection still violates (e.g., if initial violation was far), clip
                mutant = np.clip(mutant, lb, ub)

                # Crossover: binomial
                j_rand = np.random.randint(0, dim)
                trial = np.where(np.random.rand(dim) < self.CR, mutant, pop[i])
                # Ensure at least one dimension from mutant
                trial[j_rand] = mutant[j_rand]

                # Evaluate trial
                trial_fitness = func(trial)
                rem_eval -= 1

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    # Update best if improved
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        return best_x, best_y
