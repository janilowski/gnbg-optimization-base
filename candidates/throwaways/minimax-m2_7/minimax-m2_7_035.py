# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) optimizer for continuous black‑box minimization.
# Search state: Population of candidate solutions maintained as a 2‑D NumPy array.
# Candidate generation: For each target vector, three distinct parents are selected
#   and combined via differential mutation (donor = x_r1 + F*(x_r2‑x_r3)).
#   Binomial crossover mixes donor and target dimensions according to a random CR.
# Selection and replacement: Greedy (μ+λ) selection – keep the better of trial vs. target.
# Adaptation: No explicit adaptation of internal parameters; F and CR are randomly
#   sampled each generation to sustain diversity.
# Exploration mechanisms: Differential mutation introduces large exploratory steps;
#   the population spreads across the search space.
# Exploitation mechanisms: Selection gradually concentrates population around best
#   solutions found so far.
# Boundary handling: All trial vectors are clipped to the user‑provided bounds.
# Budget strategy: Population size is chosen relative to the evaluation budget and
#   problem dimension; never perform more than the allotted evaluations.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: Random F and CR each generation (instead of fixed)
#   increase variation without extra bookkeeping.
# Failure modes: If the budget is too small to sustain a diverse population,
#   performance degrades to random search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    Simple Differential Evolution (DE) optimizer for continuous black‑box minimization.

    The class follows the required interface:
        __init__(self, budget, dim)
        __call__(self, func) -> (best_x, best_y)

    It respects the evaluation budget, reads bounds from func.lower/func.upper or
    func.bounds.lb/func.bounds.ub, and uses only NumPy and standard library functions.
    """

    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        # Choose a population size that fits into the budget.
        # DE works best with at least 4 individuals, but we cannot exceed the budget.
        # Use a compromise: 10*dim is a typical DE population size, capped by budget/2
        # to leave room for multiple generations. Ensure we have at least 4 members
        # if the budget permits.
        max_pop = max(4, min(budget // 2, 10 * dim))
        self.np = min(max_pop, budget) if budget >= 4 else budget

    def __call__(self, func):
        """
        Run the optimizer on the given black‑box function.

        Parameters
        ----------
        func : callable
            Objective function to be minimized. It accepts a 1‑D NumPy array
            (the candidate) and returns a scalar (the function value).

        Returns
        -------
        best_x : NumPy array
            Best candidate found (approximate minimizer).
        best_y : float
            Function value at best_x (approximate minimum).
        """
        budget = self.budget
        dim = self.dim
        np_gen = self.np

        # -----------------------------------------------------------
        # 1. Obtain problem bounds
        # -----------------------------------------------------------
        lower, upper = self._get_bounds(func, dim)

        # -----------------------------------------------------------
        # 2. Initialise population (random uniform sampling)
        # -----------------------------------------------------------
        # If budget is smaller than np, we evaluate only budget points.
        pop = np.random.rand(np_gen, dim) * (upper - lower) + lower

        # Evaluate initial population
        fitness = np.empty(np_gen)
        evals = 0
        for i in range(np_gen):
            fitness[i] = func(pop[i])
            evals += 1
            if evals >= budget:
                # Not enough budget for a full population; return best found so far
                best_idx = np.argmin(fitness)
                return pop[best_idx].copy(), fitness[best_idx]

        # Track best solution encountered
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # -----------------------------------------------------------
        # 3. Differential Evolution loop
        # -----------------------------------------------------------
        # If the population is too small for DE (np < 4), fall back to pure random search.
        if np_gen < 4:
            # Simple random search for remaining evaluations
            while evals < budget:
                x = np.random.rand(dim) * (upper - lower) + lower
                f = func(x)
                evals += 1
                if f < best_y:
                    best_x = x.copy()
                    best_y = f
            return best_x, best_y

        # Number of full DE generations we can afford
        max_iters = (budget - evals) // np_gen

        for _ in range(max_iters):
            for i in range(np_gen):
                # ----- Mutation: pick three distinct indices -----
                idxs = [j for j in range(np_gen) if j != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

                # Scaling factor F uniformly sampled in [0.5, 1.0]
                F = np.random.uniform(0.5, 1.0)
                donor = pop[r1] + F * (pop[r2] - pop[r3])

                # ----- Crossover: binomial -----
                CR = np.random.uniform(0.7, 0.9)
                trial = np.empty(dim)
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if j == j_rand or np.random.rand() < CR:
                        trial[j] = donor[j]
                    else:
                        trial[j] = pop[i][j]

                # ----- Boundary handling -----
                trial = np.clip(trial, lower, upper)

                # ----- Evaluation -----
                f_trial = func(trial)
                evals += 1

                # ----- Greedy selection -----
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_y:
                        best_x = trial.copy()
                        best_y = f_trial

                # Stop if budget exhausted
                if evals >= budget:
                    return best_x, best_y

        return best_x, best_y

    @staticmethod
    def _get_bounds(func, dim):
        """
        Try to read lower/upper bounds from the function object.

        The method checks two common conventions:
            - func.lower / func.upper (attributes)
            - func.bounds.lb / func.bounds.ub (object with .lb/.ub)

        If none of those exist, a default bound of [-5, 5] is used for every
        dimension.

        Parameters
        ----------
        func : callable
            The objective function (or a wrapper exposing bounds).
        dim : int
            Number of dimensions.

        Returns
        -------
        lower, upper : NumPy arrays of shape (dim,)
            Lower and upper bound arrays.
        """
        # First try func.lower / func.upper
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower)
            upper = np.asarray(func.upper)
        # Then try func.bounds.lb / func.bounds.ub
        elif hasattr(func, 'bounds'):
            lb = getattr(func.bounds, 'lb', None)
            ub = getattr(func.bounds, 'ub', None)
            if lb is None or ub is None:
                # Fallback to default bounds
                return np.full(dim, -5.0), np.full(dim, 5.0)
            lower = np.asarray(lb)
            upper = np.asarray(ub)
        else:
            # No bound information -> default
            return np.full(dim, -5.0), np.full(dim, 5.0)

        # Ensure lower and upper are arrays broadcast to shape (dim,)
        lower = np.broadcast_to(lower, dim).copy()
        upper = np.broadcast_to(upper, dim).copy()
        return lower, upper
