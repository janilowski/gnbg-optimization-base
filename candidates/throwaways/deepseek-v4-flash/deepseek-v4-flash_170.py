import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This is a compact Differential Evolution (DE) minimizer using the rand/1/bin variant.
# Search state: A population of candidate solutions (real vectors) and their corresponding objective values.
# Candidate generation: For each population member, a mutant vector is created by adding the scaled difference of two other random distinct members to a third random member (mutant = x_r1 + F*(x_r2 - x_r3)).
# Selection and replacement: After binomial crossover, the trial vector replaces the parent if its objective value is strictly better. The global best is tracked.
# Adaptation: No adaptive parameters; F = 0.5 and CR = 0.9 are fixed.
# Exploration mechanisms: The differential mutation and random crossover enable global exploration.
# Exploitation mechanisms: The greedy selection and reliance on the current population drive local refinement. The best solution is always retained.
# Boundary handling: Candidate vectors are clipped component‑wise to the feasible interval [lb, ub].
# Budget strategy: The algorithm stops immediately when the number of function evaluations reaches the given budget. The population size is capped to not exceed the budget.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997), specifically the DE/rand/1/bin variant.
# Novelty or unusual aspects: None; it is a straightforward implementation intended for robustness and clarity.
# Failure modes: Premature convergence on multimodal landscapes, stagnation in low‑budget scenarios, sensitivity to the fixed F and CR values.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initialize the optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimization on a given black‑box function.

        Parameters
        ----------
        func : callable
            The objective function to minimize. Must provide bounds via either
            `func.lower` / `func.upper` or `func.bounds.lb` / `func.bounds.ub`.

        Returns
        -------
        best_x : np.ndarray
            Best found point.
        best_y : float
            Objective value at best_x.
        """
        # ---------- extract bounds ----------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            bounds_obj = func.bounds
            lb = np.asarray(bounds_obj.lb, dtype=float)
            ub = np.asarray(bounds_obj.ub, dtype=float)
        else:
            raise AttributeError("Cannot locate problem bounds. Expected "
                                 "`func.lower` / `func.upper` or "
                                 "`func.bounds.lb` / `func.bounds.ub`.")

        # ensure correct dimensionality
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)
        else:
            lb = lb.flatten()
            ub = ub.flatten()

        # ---------- set population size ----------
        # start with a heuristic that depends on dimension, but never exceed budget
        popsize = max(10, min(50, 5 * self.dim))
        popsize = min(self.budget, popsize)
        if popsize < 1:
            popsize = 1

        # ---------- initialisation ----------
        # uniform random within bounds
        pop = lb + np.random.uniform(size=(popsize, self.dim)) * (ub - lb)
        y = np.array([func(x) for x in pop])
        neval = popsize

        # best so far
        best_idx = np.argmin(y)
        best_x = pop[best_idx].copy()
        best_y = y[best_idx]

        # ---------- DE parameters (fixed) ----------
        F = 0.5
        CR = 0.9

        # ---------- main evolution loop ----------
        while neval < self.budget:
            for i in range(popsize):
                # choose three distinct indices, all different from i
                candidates = list(range(popsize))
                candidates.pop(i)                # remove i
                if len(candidates) < 3:
                    # degenerate case – should not happen with popsize >= 4
                    continue
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

                # mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # clip to bounds
                mutant = np.clip(mutant, lb, ub)

                # binomial crossover
                jrand = np.random.randint(self.dim)
                trial = np.where(
                    np.random.rand(self.dim) <= CR,
                    mutant,
                    pop[i]
                )
                trial[jrand] = mutant[jrand]   # ensure at least one dimension from mutant

                # evaluation
                trial_y = func(trial)
                neval += 1

                # selection (greedy)
                if trial_y < y[i]:
                    pop[i] = trial
                    y[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()

                # check budget after each evaluation
                if neval >= self.budget:
                    break

            # guard against empty generation
            if neval >= self.budget:
                break

        return best_x, best_y
