import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a simple yet effective derivative-free optimizer based on
# the Differential Evolution (DE) strategy. DE is a population-based evolutionary algorithm
# that mutates vectors by adding scaled differences between randomly chosen population members.
# Search state: The algorithm maintains a population of NP individuals (candidate solutions)
# and the current best solution found. Each individual is a point in the dim‑dimensional
# search space.
# Candidate generation: For each target vector a mutant is created by selecting three distinct
# parents, computing a scaled difference (F), and adding it to the first parent (rand/1 scheme).
# A trial vector is then built by mixing the target and mutant dimensions using a binomial
# crossover with probability CR.
# Selection and replacement: After evaluating the trial vector on the objective function, it
# replaces the target only if it is at least as good (for minimization).
# Adaptation: The algorithm uses fixed DE control parameters (F=0.8, CR=0.9) which are robust
# across a wide range of problems. Population size NP scales with dimensionality (10*dim)
# but is capped by the evaluation budget.
# Exploration mechanisms: The population diversity is maintained by the mutation scheme,
# which encourages exploration of the search space. Boundary handling is performed by
# clipping trial vectors to the feasible region.
# Exploitation mechanisms: The selection mechanism continuously replaces worse individuals
# with better candidates, gradually focusing the search around improving regions.
# Boundary handling: Vectors are clipped to the user‑provided lower/upper bounds after
# mutation and crossover.
# Budget strategy: The algorithm stops as soon as the total number of function evaluations
# reaches the supplied budget, ensuring no budget overrun.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997) and
# the “rand/1/bin” scheme used in many black‑box optimization competitions.
# Novelty or unusual aspects: The implementation is deliberately minimalistic to keep it
# readable, portable (only NumPy and standard library), and easy to integrate into
# benchmarking frameworks.
# Failure modes: For very small budgets the population may not have enough generations
# to converge; for highly rugged landscapes the fixed parameters may be suboptimal.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """
    Simple Differential Evolution optimizer for black‑box minimization.

    The class follows the required interface:
        __init__(self, budget, dim)  -> sets up the algorithm.
        __call__(self, func)         -> runs the optimizer on func and returns (best_x, best_y).

    The algorithm respects the evaluation budget, never exceeding it.
    Bounds are read from either func.lower / func.upper or func.bounds.lb / func.bounds.ub.
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = int(budget) if budget > 0 else 0
        self.dim = int(dim)

        # Population size: at least 5, at most 10*dim, but never larger than the budget.
        self.NP = max(min(int(10 * self.dim), self.budget), 5)

        # DE control parameters (robust defaults)
        self.F = 0.8      # Scaling factor for differential mutation
        self.CR = 0.9     # Crossover probability

    def __call__(self, func):
        """
        Run the optimizer on the given function.

        Parameters
        ----------
        func : callable
            A black‑box objective function that accepts a 1‑D NumPy array of length dim
            and returns a scalar (the objective value, minimization is assumed).

        Returns
        -------
        best_x : np.ndarray
            The best (lowest) solution found.
        best_y : float
            The corresponding objective value.
        """
        # If no budget, return trivial values
        if self.budget <= 0:
            return None, None

        # ------------------------------------------------------------------
        # Extract search space bounds
        # ------------------------------------------------------------------
        lower, upper = self._read_bounds(func)

        # ------------------------------------------------------------------
        # Helper to enforce bounds
        # ------------------------------------------------------------------
        def clip_to_bounds(x):
            return np.clip(x, lower, upper)

        # ------------------------------------------------------------------
        # Initialise population uniformly inside the hyper‑rectangle
        # ------------------------------------------------------------------
        pop = lower + np.random.rand(self.NP, self.dim) * (upper - lower)
        # Evaluate the initial population
        fitness = np.empty(self.NP)
        for i in range(self.NP):
            fitness[i] = func(pop[i])
        evals = self.NP

        # Track the best solution found so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = float(fitness[best_idx])

        # ------------------------------------------------------------------
        # Main evolution loop: generate and evaluate trial vectors
        # ------------------------------------------------------------------
        while evals < self.budget:
            # For each target vector in the population
            for i in range(self.NP):
                if evals >= self.budget:
                    break

                # --- Generate mutant -------------------------------------------------
                # Choose three distinct indices different from i
                indices = list(range(self.NP))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                # rand/1 mutation scheme
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = clip_to_bounds(mutant)

                # --- Binomial crossover -----------------------------------------------
                trial = pop[i].copy()
                # Ensure at least one dimension comes from the mutant
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                trial = clip_to_bounds(trial)

                # --- Evaluate trial ----------------------------------------------------
                trial_fitness = func(trial)
                evals += 1

                # --- Selection ---------------------------------------------------------
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    # Update global best if improvement
                    if trial_fitness < best_y:
                        best_x = trial.copy()
                        best_y = float(trial_fitness)

                # If budget exhausted after this evaluation, exit early
                if evals >= self.budget:
                    break

        return best_x, best_y

    @staticmethod
    def _read_bounds(func):
        """
        Read lower and upper bounds from the function object.

        The function may store bounds as:
            func.lower / func.upper   or   func.bounds.lb / func.bounds.ub

        Parameters
        ----------
        func : object
            Function (or function wrapper) that provides bounds.

        Returns
        -------
        lower, upper : np.ndarray
            1‑D arrays of length dim containing the lower and upper bounds.
        """
        # Try the simple attribute form
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        # Try the attribute that is a named tuple with .lb / .ub
        elif hasattr(func, 'bounds'):
            bounds = func.bounds
            lower = np.asarray(bounds.lb, dtype=float)
            upper = np.asarray(bounds.ub, dtype=float)
        else:
            raise ValueError(
                "Cannot determine search bounds. Provide either "
                "func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )

        if lower.shape != upper.shape:
            raise ValueError("Lower and upper bound shapes do not match.")
        return lower, upper
