import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Self‑adaptive Differential Evolution (jDE variant) for black‑box minimization.
#   Population evolves candidate solutions; each individual carries its own F (mutation scale)
#   and CR (crossover probability). These parameters are updated probabilistically during the run.
# Search state: A list of vectors (population), plus associated F and CR values, and the
#   best‑so‑far solution (scalar `best_y` and vector `best_x`).
# Candidate generation: For each target vector, pick three distinct random population members
#   (a,b,c) and create a mutant: x_m = a + F * (b - c). Then apply binomial crossover with the
#   target vector using probability CR to obtain the trial vector.
# Selection and replacement: If the trial vector yields a lower (better) function value than
#   the target, it replaces the target in the population, and the target’s F and CR are kept;
#   otherwise the target remains and its F and CR are unchanged.
# Adaptation: With probability tau1 (0.1) the mutated individual’s F is re‑sampled uniformly
#   in [0.1,0.9]; with probability tau2 (0.1) its CR is re‑sampled uniformly in [0,1].
#   This is done *before* generating the trial, so the new parameters are used immediately.
# Exploration mechanisms: Large F values (up to 0.9) produce large‑scale perturbations; high CR
#   mixes many components from the mutant, enabling broad exploration of the search space.
# Exploitation mechanisms: Selection pressure drives the population toward better‑performing
#   regions; as the population converges, F and CR often decrease (or remain low) causing finer
#   local searches.
# Boundary handling: Candidate coordinates that exceed the search domain are clipped to the
#   nearest bound (simple clamping).
# Budget strategy: Every call to `func` is counted. The algorithm stops immediately when the
#   budget of function evaluations is exhausted, even in the middle of a generation.
# Closest known influences: jDE (Brest et al., 2006) – self‑adaptive differential evolution.
# Novelty or unusual aspects: Purely conventional implementation; no restarts, no archives.
# Failure modes: May stagnate on highly multimodal landscapes if budget is small; the fixed
#   population size may be suboptimal for very high dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the algorithm with a given evaluation budget and dimensionality.

        Args:
            budget: Maximum number of function evaluations allowed.
            dim:    Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        # Population size: a simple heuristic, at least 4 and at most budget//2.
        pop_size = 4 + int(3 * np.log(dim))
        # Ensure we never allocate more than half the budget (leave room for generations)
        self.pop_size = min(pop_size, max(4, budget // 2))
        # Adaptation rates for F and CR (jDE defaults)
        self.tau1 = 0.1
        self.tau2 = 0.1
        # Pre‑allocated arrays for population, F, CR (will be initialized in __call__)
        self.pop = None
        self.f = None
        self.cr = None
        self.fitness = None
        self.best_x = None
        self.best_y = np.inf
        self.evals = 0

    def __call__(self, func):
        """
        Run the optimizer on the given black‑box function.

        Args:
            func: Callable that returns a scalar objective value for a 1‑D numpy array.
                  Must provide bounds via either `func.lower` / `func.upper` or
                  `func.bounds.lb` / `func.bounds.ub`.

        Returns:
            (best_x, best_y): tuple of the best found solution and its function value.
        """
        # ---------- read bounds ----------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float).ravel()
            ub = np.asarray(func.bounds.ub, dtype=float).ravel()
        else:
            raise AttributeError("Function must provide bounds via .lower/.upper or .bounds.lb/.bounds.ub")
        # Ensure bounds are 1‑D vectors of length dim
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)
        else:
            lb = lb.astype(float)
            ub = ub.astype(float)

        # ---------- initialise population ----------
        self.pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        self.f = np.random.uniform(0.1, 0.9, size=self.pop_size)
        self.cr = np.random.uniform(0.0, 1.0, size=self.pop_size)
        self.fitness = np.full(self.pop_size, np.inf)
        self.best_x = self.pop[0].copy()
        self.best_y = np.inf
        self.evals = 0

        # Evaluate initial population
        for i in range(self.pop_size):
            if self.evals >= self.budget:
                break
            y = func(self.pop[i])
            self.evals += 1
            self.fitness[i] = y
            if y < self.best_y:
                self.best_y = y
                self.best_x = self.pop[i].copy()

        # ---------- main loop ----------
        while self.evals < self.budget:
            for i in range(self.pop_size):
                if self.evals >= self.budget:
                    break

                # Choose three distinct random indices different from i
                idxs = [j for j in range(self.pop_size) if j != i]
                np.random.shuffle(idxs)
                a, b, c = idxs[:3]

                # Self‑adaptive update of F and CR (jDE style)
                if np.random.rand() < self.tau1:
                    self.f[i] = np.random.uniform(0.1, 0.9)
                if np.random.rand() < self.tau2:
                    self.cr[i] = np.random.uniform(0.0, 1.0)

                # Mutation (DE/rand/1)
                mutant = self.pop[a] + self.f[i] * (self.pop[b] - self.pop[c])

                # Binomial crossover
                j_rand = np.random.randint(0, self.dim)
                trial = np.empty(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.cr[i] or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = self.pop[i][j]

                # Boundary clamping
                trial = np.clip(trial, lb, ub)

                # Evaluation
                y_trial = func(trial)
                self.evals += 1

                # Selection
                if y_trial < self.fitness[i]:
                    self.pop[i] = trial
                    self.fitness[i] = y_trial
                    # The parameters F and CR are kept as they were (they contributed to success)
                    if y_trial < self.best_y:
                        self.best_y = y_trial
                        self.best_x = trial.copy()

        return self.best_x, self.best_y
