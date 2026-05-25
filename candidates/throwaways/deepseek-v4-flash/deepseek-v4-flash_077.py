import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) minimizer that works well across a range of dimensions.
# Search state: a population of candidate solutions (real vectors) and their objective values.
# Candidate generation: for each target vector, a mutant is created via DE/rand/1 (difference of two random
#   population members added to a third). Then binomial crossover blends the mutant with the target.
# Selection and replacement: greedy; trial replaces target only if it yields a lower (better) objective value.
# Adaptation: fixed control parameters (F=0.8, CR=0.9) – no online adaptation to keep the code simple.
# Exploration mechanisms: random selection of parents for mutation promotes diversity; crossover also mixes.
# Exploitation mechanisms: selection pressure drives the population toward better regions, and using the
#   current best as one parent (in rand/1 indirectly) helps focus.
# Boundary handling: trial vectors that exceed bounds are clipped (clamped) to the feasible range.
# Budget strategy: initial population is evaluated, then generations run until the evaluation budget is exhausted.
#   The population size is set as a function of dimension (10*dim) to balance diversity and convergence.
# Closest known influences: Standard Differential Evolution (DE) with rand/1/bin strategy.
# Novelty or unusual aspects: None – this is a textbook implementation, included for reliability and readability.
# Failure modes: May converge prematurely on multimodal landscapes if budget is too small relative to
#   population size; high-dimensional problems may require more evaluations than typical.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the DE minimizer.

        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # Population size: scale with dimension, but clamp to avoid exceeding budget with too few generations.
        self.popsize = min(10 * dim, max(5, budget // 2))
        # Ensure popsize is at least 4 (requirement for DE mutation).
        self.popsize = max(4, self.popsize)

        # Control parameters of DE (fixed).
        self.F = 0.8   # mutation factor
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        """
        Run the DE minimizer on the given objective function.

        Parameters
        ----------
        func : callable
            The objective function to minimize. Must have 'lower'/'upper' or
            'bounds.lb'/'bounds.ub' attributes for the search bounds.

        Returns
        -------
        best_x : ndarray
            Best found candidate vector.
        best_y : float
            Best found objective value.
        """
        # 1. Extract bounds ----------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)
        else:
            raise AttributeError("func must provide bounds via .lower/.upper or .bounds.lb/.bounds.ub")
        lb = lb.reshape(1, -1)
        ub = ub.reshape(1, -1)
        dim = self.dim
        popsize = self.popsize

        # 2. Initialize population ---------------------------------------------
        # Random uniform in [lb, ub].
        rng = np.random.default_rng()
        pop = lb + (ub - lb) * rng.random((popsize, dim))
        # Evaluate initial population.
        vals = np.array([func(p) for p in pop])
        evals = popsize  # initial evaluations used

        # Track best solution.
        best_idx = np.argmin(vals)
        best_x = pop[best_idx].copy()
        best_y = vals[best_idx]

        # 3. Main DE loop ----------------------------------------------------
        # We will perform full generations until we cannot fit a full popsize.
        while evals < self.budget:
            # How many trials can we still afford in this generation?
            remaining = self.budget - evals
            if remaining <= 0:
                break
            # Evaluate at most 'popsize' trials, but not more than remaining.
            trials_this_gen = min(popsize, remaining)

            # For each target index (0..popsize-1) we generate a trial vector.
            # But we only evaluate the first 'trials_this_gen' of them.
            for i in range(trials_this_gen):
                # Choose three distinct random indices, all != i.
                candidates = list(range(popsize))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

                # Mutation: DE/rand/1
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])

                # Binomial crossover with target pop[i]
                cross_mask = rng.random(dim) < self.CR
                # Ensure at least one component comes from mutant.
                if not np.any(cross_mask):
                    cross_mask[rng.integers(dim)] = True
                trial = np.where(cross_mask, mutant, pop[i])

                # Boundary handling: clip to bounds.
                trial = np.clip(trial, lb, ub)

                # Evaluate trial.
                trial_val = func(trial)
                evals += 1

                # Greedy selection: replace if better or equal (for diversity).
                if trial_val <= vals[i]:
                    pop[i] = trial
                    vals[i] = trial_val
                    # Update global best if necessary.
                    if trial_val < best_y:
                        best_x = trial.copy()
                        best_y = trial_val

                # Check budget after each evaluation.
                if evals >= self.budget:
                    break

        # Return best found solution.
        return best_x, best_y
