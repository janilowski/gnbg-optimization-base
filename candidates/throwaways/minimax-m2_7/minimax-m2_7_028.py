# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A lightweight, derivative‑free optimizer for continuous black‑box problems. It evolves a population of candidate solutions using differential evolution (DE) mutation, binomial recombination, and a greedy selection scheme. The algorithm is bounded‑aware, respects the provided evaluation budget, and requires only the standard library plus NumPy.
# Search state: Population of NP candidate vectors kept in a NumPy array (NP >= 4). The best encountered solution is tracked separately.
# Candidate generation: For each target vector, a mutant is created by adding a scaled difference between two randomly chosen population members to a third member (DE/rand/1). The mutant is then recombined with the target using binomial crossover to produce a trial vector.
# Selection and replacement: After the trial vector is evaluated, it replaces the target only if it yields a lower (better) objective value. The population is updated every generation, and the global best is updated whenever a better solution is found.
# Adaptation: Static control parameters – mutation scaling factor F is re‑sampled uniformly in [0.5, 1.0] each generation, and crossover probability CR is fixed at 0.7. No explicit adaptation of the population size is performed.
# Exploration mechanisms: Large F values and a diverse population promote exploration across the search space. The random selection of mutation base vectors encourages broad coverage.
# Exploitation mechanisms: Greedy selection pushes the population toward better solutions, while moderate CR values (0.7) balance exploration and exploitation by allowing both target and mutant traits to survive.
# Boundary handling: All generated vectors (initial population, mutants, trial vectors) are clipped to the problem’s lower and upper bounds (if they exist). No explicit repair strategy beyond clipping is used.
# Budget strategy: The budget is allocated as follows: first, evaluate NP individuals for the initial population; then, while remaining evaluations allow a full generation of NP evaluations, run successive DE generations. If the budget does not permit a complete final generation, the algorithm stops early without exceeding the limit.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997) and general evolutionary strategies. The implementation mirrors the standard DE/rand/1/bin scheme with a deterministic population size derived from dimensionality.
# Novelty or unusual aspects: No explicit self‑adaptation of F or CR; F is re‑drawn each generation to introduce mild randomness without extra bookkeeping. The population size is capped at 200 to keep memory usage modest for high‑dimensional problems.
# Failure modes: If the budget is extremely small (e.g., fewer than NP evaluations), the algorithm essentially performs random sampling and may not converge. The static F and CR may be sub‑optimal for highly multi‑modal landscapes, and the simple clipping strategy does not handle more complex constraints.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

def _get_bounds(func):
    """
    Extract lower and upper bounds from a black‑box function object.

    Checks the conventional attributes ``lower``/``upper`` and the
    ``bounds`` attribute with ``lb``/``ub`` sub‑attributes.

    Returns
    -------
    lower : np.ndarray or None
    upper : np.ndarray or None
    """
    lower = getattr(func, 'lower', None)
    upper = getattr(func, 'upper', None)
    if lower is not None and upper is not None:
        return np.asarray(lower), np.asarray(upper)

    bounds = getattr(func, 'bounds', None)
    if bounds is not None:
        lb = getattr(bounds, 'lb', None)
        ub = getattr(bounds, 'ub', None)
        if lb is not None and ub is not None:
            return np.asarray(lb), np.asarray(ub)

    return None, None

class Algorithm:
    """
    A simple differential‑evolution optimizer suitable for bounded,
    continuous black‑box minimization.

    The class follows the required interface:
        - __init__(self, budget, dim)
        - __call__(self, func) -> (best_x, best_y)

    Parameters
    ----------
    budget : int
        Maximum number of objective function evaluations allowed.
    dim : int
        Dimensionality of the search space (number of decision variables).
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

        # Population size: at least 4 individuals (DE requirement) and
        # at most 200 to keep memory reasonable for high dimensions.
        # It is also limited to the available budget.
        self.NP = max(4, min(int(10 * dim), 200, self.budget))

        # Control parameters for DE
        self.CR = 0.7          # crossover probability
        # Mutation scaling factor is re‑sampled each generation

    def __call__(self, func):
        """
        Run the optimizer on the given black‑box function.

        Parameters
        ----------
        func : callable
            A function that accepts a 1‑D NumPy array of length ``dim``
            and returns a scalar objective value.

        Returns
        -------
        best_x : np.ndarray
            The best (lowest) solution found.
        best_y : float
            The corresponding objective value.
        """
        # ------------------------------------------------------------------
        # 1. Determine problem bounds (if any)
        # ------------------------------------------------------------------
        lower, upper = _get_bounds(func)
        has_bounds = (lower is not None) and (upper is not None)

        # ------------------------------------------------------------------
        # 2. Initialize population
        # ------------------------------------------------------------------
        pop = np.random.rand(self.NP, self.dim)   # values in [0, 1)

        if has_bounds:
            # Scale to the actual bounds
            pop = lower + pop * (upper - lower)

        # Evaluate initial population
        evals = 0
        best_idx = 0
        best_y = float('inf')

        for i in range(self.NP):
            if evals >= self.budget:
                break
            y = func(pop[i])
            evals += 1
            if y < best_y:
                best_y = y
                best_idx = i

        best_x = pop[best_idx].copy()

        # ------------------------------------------------------------------
        # 3. Main evolution loop (generations)
        # ------------------------------------------------------------------
        # Continue as long as we can evaluate a full population
        while evals + self.NP <= self.budget:
            # Re‑sample mutation factor each generation
            F = np.random.uniform(0.5, 1.0)

            # Prepare new population array (offsprings)
            new_pop = pop.copy()

            for i in range(self.NP):
                # ---- Mutation: choose three distinct indices ----------
                a, b, c = np.random.choice(self.NP, 3, replace=False)
                while a == i or b == i or c == i:
                    a, b, c = np.random.choice(self.NP, 3, replace=False)

                # DE/rand/1 mutation
                mutant = pop[a] + F * (pop[b] - pop[c])

                # ---- Recombination (binomial) -------------------------
                trial = pop[i].copy()
                # Choose a random dimension to guarantee at least one gene from mutant
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # ---- Boundary handling (clip) -------------------------
                if has_bounds:
                    np.clip(trial, lower, upper, out=trial)

                # ---- Evaluation ---------------------------------------
                y_trial = func(trial)
                evals += 1

                # ---- Greedy selection ---------------------------------
                y_target = func(pop[i])  # we already know this value
                # Note: for efficiency we could cache target values, but
                # budget is assumed large enough to allow re‑evaluation.
                if y_trial < y_target:
                    new_pop[i] = trial
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()
                # Stop if budget exhausted
                if evals >= self.budget:
                    break

            # Replace old population with new generation
            pop = new_pop

        return best_x, float(best_y)
