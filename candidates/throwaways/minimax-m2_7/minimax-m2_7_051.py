# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, population‑based stochastic optimizer based on
#          Differential Evolution (DE). DE evolves a set of candidate solutions (the population)
#          by mutating them with scaled differences of other members and then selecting the
#          better of the trial and target vectors. The algorithm is deliberately simple, using
#          only NumPy and the Python standard library, and respects the provided evaluation
#          budget without exceeding it.
#
# Search state: The optimizer maintains a population of NP vectors in the decision space, the
#               current best solution found, and a counter of used function evaluations.
#
# Candidate generation: For each target vector a mutant is created by picking three distinct
#                      random members of the population, computing a scaled difference (b‑c)
#                      multiplied by a mutation factor F, and adding it to the first vector a.
#                      A binomial crossover then mixes coordinates of the target and mutant to
#                      form a trial vector.
#
# Selection and replacement: After the trial vector is evaluated, it replaces the target vector
#                            in the population if its objective value is not larger (i.e., it is
#                            better or equal for minimization). The global best solution is
#                            updated whenever a better individual appears.
#
# Adaptation: The implementation uses fixed control parameters (mutation factor F and crossover
#             probability CR) which are common default choices in DE. No explicit adaptation of
#             these parameters is performed, but the population size is scaled with dimension
#             and available budget to give a balance between exploration and exploitation.
#
# Exploration mechanisms: The mutation step with factor F adds a random, scaled direction,
#                        encouraging the population to explore new regions. The crossover
#                        probability CR introduces variability, preventing the population from
#                        converging too quickly.
#
# Exploitation mechanisms: Selection (keeping the better of target and trial) drives the
#                          population toward promising regions. The best‑so‑far solution is
#                          retained and returned, focusing the search on the currently known
#                          best area.
#
# Boundary handling: All vectors are clipped to the feasible bounds (lower/upper limits) after
#                    mutation and before evaluation, ensuring the optimizer never proposes a
#                    point outside the allowed domain.
#
# Budget strategy: The algorithm counts each function evaluation and stops as soon as the
#                  counter reaches the supplied budget. Initial population size is limited to
#                  budget‑1 to leave room for subsequent generations; if the budget is very
#                  small the algorithm falls back to evaluating random points only.
#
# Closest known influences: The code closely follows the classic Differential Evolution
#                           algorithm described by Storn and Price (1997). It shares the same
#                           core ideas of mutation, crossover, and selection used in many
#                           evolutionary algorithms for continuous domains.
#
# Novelty or unusual aspects: The implementation is deliberately minimal and self‑contained,
#                             making it easy to embed in benchmarking frameworks. It avoids
#                             external dependencies and uses only standard library features.
#
# Failure modes: If the evaluation budget is extremely low (e.g., fewer than three evaluations)
#                the algorithm cannot generate meaningful mutants and will merely return the
#                best among a few random samples. For very small populations (NP < 3) the DE
#                mutation scheme is replaced by simple random perturbations.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple population‑based optimizer using Differential Evolution.
    Designed for continuous black‑box minimization under a strict evaluation budget.
    """

    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the decision space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimizer on the given objective function.

        Parameters
        ----------
        func : callable
            A black‑box objective that accepts a 1‑D NumPy array of length `dim`
            and returns a scalar (the objective value).

        Returns
        -------
        best_x : np.ndarray
            The decision vector that achieved the smallest observed objective value.
        best_y : float
            The corresponding objective value (best_y = func(best_x)).
        """
        dim = self.dim
        budget = self.budget

        # ------------------------------------------------------------
        # Retrieve problem bounds (lower, upper) from the function object.
        # The harness is expected to expose either `.lower`/`.upper` or
        # `.bounds.lb`/`.bounds.ub`. If none are found, we default to [0,1]^dim.
        # ------------------------------------------------------------
        lb, ub = self._read_bounds(func, dim)

        # ------------------------------------------------------------
        # Population size heuristics: try to keep a reasonably sized
        # population while respecting the budget. At least 3 individuals
        # are required for DE's mutation scheme.
        # ------------------------------------------------------------
        if budget < 3:
            # Not enough evaluations for a meaningful population; just
            # evaluate random points and return the best.
            best_x = None
            best_y = np.inf
            for _ in range(budget):
                x = np.random.uniform(lb, ub, size=dim)
                y = func(x)
                if y < best_y:
                    best_x, best_y = x.copy(), y
            return best_x, best_y

        # Choose a population size that scales with dimension, but cap it.
        NP = min(10 * dim, budget - 1)
        NP = max(NP, 3)  # ensure at least three individuals for DE

        # ------------------------------------------------------------
        # Initialise the population uniformly inside the bounds.
        # ------------------------------------------------------------
        pop = np.random.uniform(lb, ub, size=(NP, dim))

        # Evaluate initial population.
        fitness = np.empty(NP, dtype=float)
        evals = 0
        for i in range(NP):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1

        # Identify the best individual seen so far.
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = float(fitness[best_idx])

        # ------------------------------------------------------------
        # Differential Evolution control parameters.
        # F: mutation scaling factor (exploration step size)
        # CR: crossover probability (balance between target and mutant)
        # ------------------------------------------------------------
        F = 0.5
        CR = 0.7

        # ------------------------------------------------------------
        # Main evolution loop.
        # ------------------------------------------------------------
        while evals < budget:
            # Iterate over each target vector.
            for i in range(NP):
                if evals >= budget:
                    break

                # ----- Mutation -----
                # Choose three distinct indices different from i.
                indices = list(range(NP))
                indices.remove(i)
                a_idx, b_idx, c_idx = np.random.choice(indices, 3, replace=False)

                a = pop[a_idx]
                b = pop[b_idx]
                c = pop[c_idx]

                # Compute mutant vector and clip to bounds.
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, lb, ub)

                # ----- Crossover -----
                # Binomial crossover: copy each coordinate from mutant with probability CR,
                # otherwise keep the target coordinate.
                j_rand = np.random.randint(dim)  # ensure at least one coordinate comes from mutant
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])

                # ----- Evaluation -----
                trial_f = func(trial)
                evals += 1

                # ----- Selection -----
                if trial_f <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f <= best_y:
                        best_x = trial.copy()
                        best_y = trial_f

                # Stop early if budget exhausted.
                if evals >= budget:
                    break

        return best_x, best_y

    # --------------------------------------------------------------------
    # Helper method to read lower/upper bounds from the function object.
    # --------------------------------------------------------------------
    def _read_bounds(self, func, dim):
        """
        Extract lower and upper bounds from `func`.

        Expected attributes (tried in order):
        1. `func.lower` and `func.upper`
        2. `func.bounds.lb` and `func.bounds.ub`
        3. None – defaults to 0 and 1 for every dimension.

        Parameters
        ----------
        func : callable
            Objective function with possible bound attributes.
        dim : int
            Dimensionality (used only for default bounds).

        Returns
        -------
        lb, ub : np.ndarray
            Arrays of shape (dim,) containing the lower and upper limits.
        """
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: unit hypercube.
            lb = np.zeros(dim, dtype=float)
            ub = np.ones(dim, dtype=float)

        # Ensure the bounds are arrays of length `dim`.
        if lb.shape != (dim,):
            lb = np.broadcast_to(lb, (dim,)).copy()
        if ub.shape != (dim,):
            ub = np.broadcast_to(ub, (dim,)).copy()

        return lb, ub
