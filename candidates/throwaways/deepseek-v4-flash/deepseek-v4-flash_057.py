# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) minimizer designed for the GNBG black‑box benchmark. It uses the standard DE/rand/1/bin variant with a fixed population size, adaptive reflection‑based boundary handling, and runs until the evaluation budget is exhausted.
# Search state: A population of candidate solutions (real‑valued vectors) and their corresponding function values. The current best solution (and its value) is tracked throughout the run.
# Candidate generation: For each population member, a trial vector is created by adding a scaled difference of two other randomly selected population members to a third distinct member (mutation), then applying binomial crossover with the original member.
# Selection and replacement: Greedy – the trial vector replaces the target vector if its objective value is strictly lower (minimization). The best solution is updated whenever a better candidate is found.
# Adaptation: No adaptation of DE parameters (F=0.8, CR=0.9) or population size during the run. The algorithm relies on a fixed parameter set that works robustly across most problems.
# Exploration mechanisms: The mutation operator (difference of two random vectors) provides global exploration, especially in early generations. The binomial crossover blends trial and target components, promoting diversity.
# Exploitation mechanisms: Greedy selection and the use of the population’s difference vectors create implicit local refinement. As the population converges, differences shrink, leading to finer search.
# Boundary handling: Reflective (bounce‑back) repair: if a trial coordinate falls outside the feasible box, it is reflected symmetrically inward, preserving diversity and respecting bounds.
# Budget strategy: The budget is divided into an initial evaluation of the whole population, followed by sequential generations until the remaining evaluations cannot complete a full generation. Unused evaluations are lost; the algorithm stops immediately when the budget is reached.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997), specifically the DE/rand/1/bin strategy.
# Novelty or unusual aspects: None – the implementation is a straightforward, parameter‑fixed classic DE.
# Failure modes: May stagnate on highly multimodal or deceptive landscapes if the population loses diversity prematurely. Fixed parameters (F, CR) may be suboptimal for some problem instances. Performance degrades when the budget is too small to evolve the population past the initial random search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the DE optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations.
        dim : int
            Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # DE parameters (fixed)
        self.F = 0.8          # mutation scaling factor
        self.CR = 0.9         # crossover probability

        # Population size: at least 4, at most budget // 2, and scaled with dimension
        self.NP = max(4, min(50, budget // 2, 10 * dim))

    def _reflect(self, x, lb, ub):
        """Reflect coordinate into [lb, ub] by bouncing inward."""
        x = np.where(x < lb, 2 * lb - x, x)
        x = np.where(x > ub, 2 * ub - x, x)
        # In case of multiple reflections (e.g., far outside), clip as a fallback
        x = np.clip(x, lb, ub)
        return x

    def __call__(self, func):
        """
        Run the DE minimizer.

        Parameters
        ----------
        func : callable
            The objective function. Must expose either `lower`/`upper` attributes
            or `bounds.lb`/`bounds.ub` to retrieve the search bounds.

        Returns
        -------
        best_x : np.ndarray
            Best found point (1-D array of length dim).
        best_y : float
            Best objective value.
        """
        # --- 1. Read bounds ---
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective function must provide bounds via lower/upper or bounds.lb/bounds.ub")

        lb = np.broadcast_to(lb, (self.dim,))
        ub = np.broadcast_to(ub, (self.dim,))

        # --- 2. Initialize population ---
        NP = self.NP
        pop = np.random.uniform(lb, ub, size=(NP, self.dim))   # shape (NP, D)
        fitness = np.full(NP, np.inf)

        # Evaluate initial population
        evals = 0
        for i in range(NP):
            fitness[i] = func(pop[i])
            evals += 1
            if evals > self.budget:
                break

        # Track best solution
        best_idx = np.argmin(fitness[:NP])  # in case break early
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # --- 3. Main DE loop ---
        generation = 0
        while evals < self.budget:
            generation += 1
            # For each target vector, generate a trial vector
            for i in range(NP):
                if evals >= self.budget:
                    break

                # Choose three distinct random indices, different from i
                candidates = list(range(NP))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutation: v = pop[r1] + F * (pop[r2] - pop[r3])
                v = pop[r1] + self.F * (pop[r2] - pop[r3])

                # Binomial crossover
                j_rand = np.random.randint(self.dim)
                trial = pop[i].copy()
                for d in range(self.dim):
                    if np.random.rand() < self.CR or d == j_rand:
                        trial[d] = v[d]

                # Boundary reflection
                trial = self._reflect(trial, lb, ub)

                # Evaluate trial
                trial_y = func(trial)
                evals += 1

                # Greedy selection
                if trial_y < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_y
                    if trial_y < best_y:
                        best_x = trial.copy()
                        best_y = trial_y

        return best_x, best_y
