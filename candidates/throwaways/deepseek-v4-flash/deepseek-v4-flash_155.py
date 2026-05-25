import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A classic Differential Evolution (DE/rand/1/bin) optimizer for
# black-box minimization. It is simple, robust, and works across dimensions.
# Search state: The population (NP individuals, each with dim parameters) and
# their fitness values. The best individual and its fitness are tracked.
# Candidate generation: For each target vector, a mutant vector is created by
# adding the scaled difference of two random population vectors to a third.
# Then binomial crossover mixes the mutant with the target to produce a trial.
# Selection and replacement: The trial replaces the target if its fitness is
# better (or equal, but greedy selection is used for minimization).
# Adaptation: No adaptive parameters; scaling factor F and crossover rate CR
# are fixed. Population size is chosen based on budget and dimension.
# Exploration mechanisms: High F and CR encourage diversity; random initial
# population covers the search space; mutation explores new directions.
# Exploitation mechanisms: As generations progress, the population converges;
# the difference vectors shrink, focusing the search near current best.
# Boundary handling: Parameters are clipped to the bounds after mutation and
# crossover.
# Budget strategy: The budget is exhausted by limiting the number of function
# evaluations (each generation uses up to NP evaluations). The loop stops as
# soon as the budget is reached, even mid-generation.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: A very straightforward implementation without any
# bells and whistles; designed for clarity and ease of understanding.
# Failure modes: May struggle with highly multimodal or deceptive landscapes;
# fixed parameters may not be optimal across all problems; may converge
# prematurely if F is too small or population too small.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the Differential Evolution optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # Population size – aim for a reasonable trade-off between diversity
        # and generations. At most 50, at least 4, and scaled to budget.
        self.NP = max(4, min(50, budget // 10))
        # Ensure that we can run at least one generation.
        self.NP = min(self.NP, budget)  # avoid more than budget individuals
        # Differential evolution parameters (fixed, well‑known defaults)
        self.F = 0.8   # scaling factor
        self.CR = 0.9  # crossover rate

    def __call__(self, func):
        """
        Run the DE optimizer on the given function.

        Parameters
        ----------
        func : callable
            Objective function to minimize. Must have attributes `lower` and
            `upper` (or `bounds.lb` and `bounds.ub`) providing the search
            boundaries as 1‑D arrays of length dim.

        Returns
        -------
        best_x : ndarray
            Best solution found.
        best_y : float
            Best objective value found.
        """
        # --- 1. Read bounds ---
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # Assume a structure with .lb and .ub (like from scipy or COCO)
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("The function must provide 'lower/upper' "
                                 "or 'bounds.lb/bounds.ub'")
        dim = self.dim
        NP = self.NP
        budget = self.budget

        # --- 2. Initialise population uniformly within bounds ---
        pop = np.random.uniform(lb, ub, size=(NP, dim))
        # Evaluate initial population
        fits = np.array([func(x) for x in pop])
        evals = NP
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_y = fits[best_idx]

        # --- 3. Main DE loop ---
        while evals < budget:
            new_pop = pop.copy()      # will be updated in‑place
            # Generate trial vectors for each target vector
            for i in range(NP):
                # Choose three distinct random indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                r = np.random.choice(candidates, size=3, replace=False)
                a, b, c = pop[r[0]], pop[r[1]], pop[r[2]]

                # Mutation: DE/rand/1
                mutant = a + self.F * (b - c)

                # Crossover: binomial (uniform)
                mask = np.random.rand(dim) < self.CR
                # At least one component must be taken from mutant
                if not np.any(mask):
                    mask[np.random.randint(dim)] = True
                trial = np.where(mask, mutant, pop[i])

                # Boundary handling – clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate only if we still have budget
                if evals >= budget:
                    break
                trial_fit = func(trial)
                evals += 1

                # Selection: greedy for minimization
                if trial_fit <= fits[i]:
                    new_pop[i] = trial
                    fits[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

                # Update best (if already updated, no need to check all)
                # (tracked directly above)

            pop = new_pop
            # Optional: re‑evaluate best if needed (already done incrementally)

        return best_x, best_y
