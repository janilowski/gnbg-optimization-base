# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary:
#   This module implements a simple Differential Evolution (DE) optimizer for continuous
#   black‑box minimization. DE maintains a population of candidate solutions that evolve
#   through mutation, crossover, and selection in order to locate better points in the
#   search space. The algorithm is population‑based, stochastic, and does not require
#   gradient information.
#
# Search state:
#   - Population array (NP × dim) holding current candidate vectors.
#   - Fitness array (NP) storing the objective value for each candidate.
#   - Integer counters for total evaluations and index of the best solution found so far.
#
# Candidate generation:
#   - Mutation: a donor vector is created from the current best solution plus a scaled
#     difference of two randomly chosen, distinct population members: donor = best + F*(r1‑r2).
#   - Crossover: a trial vector is built by mixing the donor and the target vector using a
#     binomial scheme (each dimension is taken from the donor with probability CR, otherwise
#     from the target). To guarantee at least one donor component, one randomly chosen
#     dimension is always taken from the donor.
#
# Selection and replacement:
#   - Deterministic (μ+μ) selection: if the trial's objective value is less than or equal
#     to the target's value, the target is replaced by the trial; otherwise the target
#     remains unchanged. This preserves the population size.
#
# Adaptation:
#   - No explicit adaptation of control parameters. The scaling factor F and crossover
#     probability CR are fixed (default 0.5 and 0.7). The population size is chosen as
#     NP = max(10, 5·dim) and kept constant throughout the run.
#
# Exploration mechanisms:
#   - Random sampling of the population to generate diverse mutation directions.
#   - Recombination introduces variability and prevents premature convergence.
#   - Large initial population relative to dimensionality helps maintain diversity.
#
# Exploitation mechanisms:
#   - Bias toward the currently known best solution via best‑based mutation, driving the
#     population toward promising regions.
#   - Steady selection pressure (replace only when improvement) preserves good solutions.
#
# Boundary handling:
#   - After mutation and before evaluation, all components of donor/trial vectors are
#     clipped to the problem’s lower and upper bounds, preventing domain violations.
#
# Budget strategy:
#   - The budget (maximum number of objective evaluations) is respected by counting each
#     evaluation and exiting as soon as the counter reaches the limit. The initial
#     population is evaluated up to the remaining budget, and each subsequent trial costs
#     exactly one evaluation. The algorithm never performs extra evaluations beyond the
#     supplied budget.
#
# Closest known influences:
#   - The implementation follows the classic Differential Evolution algorithm (Storn & Price,
#     1997), specifically the “best/1/bin” variant where the best individual is used as
#     the base vector and a binomial crossover is applied.
#
# Novelty or unusual aspects:
#   - Using a static population size and fixed control parameters keeps the code compact
#     and easy to understand. The algorithm is intentionally simple to serve as a robust
#     baseline across a wide range of dimensions without requiring external libraries.
#
# Failure modes:
#   - Static parameters may be sub‑optimal for highly multi‑modal or ill‑scaled landscapes.
#   - The population size grows linearly with dimension, which may become memory‑intensive
#     for very high dimensions (though it remains manageable for typical use cases).
#   - Lack of adaptive step‑size or covariance adaptation can lead to slower convergence on
#     certain problems compared to more advanced methods (e.g., CMA‑ES).
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    """
    Simple Differential Evolution (DE) optimizer for continuous black‑box minimization.

    The class follows the required interface:
        __init__(self, budget, dim)
        __call__(self, func) -> (best_x, best_y)

    Parameters
    ----------
    budget : int
        Maximum number of objective function evaluations allowed.
    dim : int
        Dimensionality of the search space.
    """

    def __init__(self, budget: int, dim: int):
        """
        Initialize the optimizer with the given evaluation budget and problem dimension.

        Args:
            budget (int): Total number of evaluations the algorithm may perform.
            dim (int): Number of variables (dimensions) of the problem.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the DE optimizer on the given objective function.

        Args:
            func: A callable that accepts a NumPy array (candidate) and returns a scalar
                  objective value. The function must expose the problem bounds either via
                  attributes `lower` / `upper` or via a `bounds` object with attributes
                  `lb` / `ub`.

        Returns:
            tuple: (best_x, best_y) where best_x is the best solution found (NumPy array)
                   and best_y is its corresponding objective value (float).
        """
        # ------------------------------------------------------------------
        # Extract problem bounds (lower/upper limits)
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: if no bounds are provided, use a wide default range.
            lb = np.full(self.dim, -10.0)
            ub = np.full(self.dim, 10.0)

        # Ensure bounds are NumPy arrays of shape (dim,)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        # ------------------------------------------------------------------
        # Configure DE control parameters
        # ------------------------------------------------------------------
        # Population size; make it even to simplify loops.
        NP = max(10, 5 * self.dim)
        if NP % 2 != 0:
            NP += 1

        # Scaling factor for mutation (F) and crossover probability (CR).
        F = 0.5
        CR = 0.7

        # ------------------------------------------------------------------
        # Initialize population within the bounds
        # ------------------------------------------------------------------
        # Generate NP candidates uniformly in the hyper‑rectangle.
        pop = lb + (ub - lb) * np.random.rand(NP, self.dim)

        # ------------------------------------------------------------------
        # Evaluate initial population (respect the budget)
        # ------------------------------------------------------------------
        fitness = np.empty(NP, dtype=float)
        evals = 0
        best_idx = 0

        # Evaluate the first candidate.
        fitness[0] = func(pop[0])
        evals += 1
        best_y = fitness[0]

        # Evaluate remaining candidates as long as we stay within budget.
        for i in range(1, NP):
            if evals >= self.budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_idx = i

        # If the budget is exhausted after the initial评估, return what we have.
        if evals >= self.budget:
            return pop[best_idx].copy(), best_y

        # ------------------------------------------------------------------
        # Main DE loop: generate and evaluate new candidate vectors
        # ------------------------------------------------------------------
        while evals < self.budget:
            # Iterate over each target vector in the population.
            for i in range(NP):
                # Ensure we do not exceed the budget.
                if evals >= self.budget:
                    break

                # --- Mutation ---------------------------------------------------------
                # Select three distinct indices different from the target index i.
                indices = list(range(NP))
                indices.remove(i)
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)

                # Form the donor vector using the current best solution as base.
                donor = pop[best_idx] + F * (pop[r1] - pop[r2])

                # Clip the donor to respect the problem bounds.
                donor = np.clip(donor, lb, ub)

                # --- Crossover ---------------------------------------------------------
                # Create trial vector by binomial crossover.
                # Choose a random dimension to guarantee at least one component from donor.
                j_rand = np.random.randint(self.dim)
                # Each dimension is taken from donor with probability CR.
                mask = np.random.rand(self.dim) < CR
                trial = np.where(mask, donor, pop[i])
                trial[j_rand] = donor[j_rand]

                # --- Evaluation -------------------------------------------------------
                y_trial = func(trial)
                evals += 1

                # --- Selection --------------------------------------------------------
                if y_trial <= fitness[i]:
                    # Replace target with trial.
                    pop[i] = trial
                    fitness[i] = y_trial
                    # Update global best if necessary.
                    if y_trial < best_y:
                        best_y = y_trial
                        best_idx = i

        # ------------------------------------------------------------------
        # Return the best solution found
        # ------------------------------------------------------------------
        return pop[best_idx].copy(), float(best_y)
