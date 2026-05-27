import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A classic Differential Evolution (DE/rand/1/bin) optimizer. It maintains a population of candidate solutions and iteratively generates trial vectors via differential mutation and binomial crossover, replacing a parent if the trial yields a lower objective value. The algorithm terminates when the evaluation budget is exhausted.
# Search state: A list of numpy arrays (population) and corresponding objective values. Global best solution and value are tracked.
# Candidate generation: For each parent, three distinct random population members (excluding the current one) are selected. A mutant vector is created as base + F * (difference). Base is the first random member. Then binomial crossover mixes the mutant with the parent with probability CR, creating a trial vector. F and CR are fixed parameters.
# Selection and replacement: If the trial vector yields a lower objective value (minimization) than the parent, it replaces the parent in the population. Otherwise, the parent persists.
# Adaptation: No adaptation is used; parameters F and CR are constant.
# Exploration mechanisms: Mutation uses scaled differences between random population members, promoting diversity. Crossover combines information from parent and mutant. Random initialization covers the search space.
# Exploitation mechanisms: As the population converges, mutation step sizes shrink naturally because differences become smaller. The greedy replacement (only if better) drives convergence toward promising regions.
# Boundary handling: Trial vectors are clipped component-wise to the domain bounds [low, high] after generation.
# Budget strategy: The algorithm respects the budget by counting each function evaluation. It runs generations until no more evaluations are available for the next full generation (or stops early if budget exhausted mid-generation). It may leave some budget unused if not enough for a complete generation, but it will not exceed.
# Closest known influences: Standard DE/rand/1/bin as described by Storn & Price (1997).
# Novelty or unusual aspects: None; this is a textbook implementation.
# Failure modes: May get stuck in local optima on multimodal functions if population collapses prematurely. Fixed parameters might be suboptimal for certain landscapes. For very high dimensions (e.g., >100), performance degrades.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """Initialize DE optimizer with problem dimension and evaluation budget."""
        self.budget = budget
        self.dim = dim

        # Algorithm parameters
        self.F = 0.5          # mutation factor
        self.CR = 0.9         # crossover rate

        # Population size: scale with sqrt(dim), but cap to ensure at least one generation after initialization.
        self.pop_size = max(5, min(int(10 * np.sqrt(dim)), budget // 2))
        # Ensure population size never exceeds budget
        self.pop_size = min(self.pop_size, budget)

        # Internal state
        self.population = None   # list of 1-d arrays
        self.fitness = None      # list of floats
        self.best_x = None
        self.best_y = float('inf')
        self.eval_count = 0

    def __call__(self, func) -> tuple:
        """Run the optimizer on a given function, returning (best_x, best_y)."""
        # Retrieve bounds from the function object
        try:
            lower = func.lower
            upper = func.upper
        except AttributeError:
            try:
                lower = func.bounds.lb
                upper = func.bounds.ub
            except AttributeError:
                # Fallback: assume default bounds (e.g., [-100, 100])
                lower = np.full(self.dim, -100.0)
                upper = np.full(self.dim, 100.0)
        try:
            dim = func.dimension
        except AttributeError:
            dim = self.dim

        # Ensure dimensions match
        if dim != self.dim:
            raise ValueError(f"Function dimension {dim} does not match algorithm dimension {self.dim}")

        # Convert bounds to numpy arrays
        low = np.atleast_1d(np.asarray(lower, dtype=float)).flatten()
        high = np.atleast_1d(np.asarray(upper, dtype=float)).flatten()

        # Initialize population
        self.population = [np.random.uniform(low, high, dim).astype(float) for _ in range(self.pop_size)]
        self.fitness = [float('inf')] * self.pop_size

        # Evaluate initial population
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = func(self.population[i])
            self.eval_count += 1
            self.fitness[i] = y
            if y < self.best_y:
                self.best_y = y
                self.best_x = self.population[i].copy()

        # Main DE loop
        while self.eval_count < self.budget:
            # For each individual, generate a trial vector
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Choose three distinct indices different from i
                candidates = [j for j in range(self.pop_size) if j != i]
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Current parent
                parent = self.population[i]

                # Mutant (DE/rand/1)
                mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])

                # Binomial crossover
                trial = np.where(np.random.rand(dim) < self.CR, mutant, parent)

                # Clip to bounds
                trial = np.clip(trial, low, high)

                # Evaluate trial
                y_trial = func(trial)
                self.eval_count += 1

                # Selection (greedy)
                if y_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = y_trial
                    if y_trial < self.best_y:
                        self.best_y = y_trial
                        self.best_x = trial.copy()

        return self.best_x, self.best_y
