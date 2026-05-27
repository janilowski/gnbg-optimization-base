import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple Differential Evolution (DE) variant with dither and exponential crossover,
#           adapted for black-box minimization with a strict evaluation budget.
# Search state: A fixed-size population of candidate solutions (vectors), each with an associated
#               fitness value, initialized uniformly within the search domain.
# Candidate generation: For each target vector, a mutant is produced by adding a scaled difference
#                       of two random population vectors to a third base vector (DE/rand/1).
# Selection and replacement: Each trial vector replaces the target vector in the next generation
#                            only if it has strictly better (lower) fitness.
# Adaptation: The scale factor F is randomly chosen per generation from [0.5, 1.0) (dither) and
#             the crossover rate CR is fixed at 0.9. Population size is set by budget.
# Exploration mechanisms: Random parent selection for mutation, dither scales, and exponential
#                         crossover with high CR maintain diversity.
# Exploitation mechanisms: Difference vectors pull toward better regions; greedy selection polishes.
# Boundary handling: New coordinates are reflected back into the allowed range after mutation and
#                    crossover (bounce-back).
# Budget strategy: The budget is split evenly across generations (pop_size = budget // 3, at least 5).
#                  The algorithm runs exactly that many generations, evaluating exactly population
#                  size many candidates per generation.
# Closest known influences: Classic DE/rand/1/bin with a deterministic CR and dithering F.
# Novelty or unusual aspects: Exponentially distributed crossover segment length (instead of binomial)
#                             for slightly different linkage behavior.
# Failure modes: Stagnation on highly multimodal landscapes with no crossover adaptation; may
#                converge prematurely if population size is too small relative to dimension.
# ALGORITHM_ANALYSIS_NOTE_END


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

        # Population size: use a fraction of budget, at least 5.
        # Ensures we have enough generations for search.
        self.pop_size = max(5, budget // 3)
        self.generations = budget // self.pop_size  # number of full generations
        # Remaining evaluations (if any) are unused to stay under budget.

        # Control parameters
        self.F_low = 0.5
        self.F_high = 1.0
        self.CR = 0.9

        # Population arrays
        self.pop = None   # shape (pop_size, dim)
        self.fitness = None  # shape (pop_size,)

        # Boundaries (filled in __call__)
        self.lb = None
        self.ub = None

    def _bounce_back(self, x):
        """Reflect out-of-bounds coordinates back into [lb, ub]."""
        lower = self.lb
        upper = self.ub
        # Reflect: if x < lb, new = lb + (lb - x)
        #          if x > ub, new = ub - (x - ub)
        x = np.where(x < lower, 2 * lower - x, x)
        x = np.where(x > upper, 2 * upper - x, x)
        # Clamp in case of extreme reflection overshoot (should rarely happen)
        return np.clip(x, lower, upper)

    def __call__(self, func):
        """
        Run the optimizer on a given objective function.

        Parameters
        ----------
        func : callable
            Objective function with attributes lower/upper or bounds.lb/bounds.ub.

        Returns
        -------
        best_x : numpy.ndarray
            Best found solution.
        best_y : float
            Best found objective value.
        """
        # Read bounds (robust to different attribute names)
        try:
            self.lb = np.array(func.lower, dtype=float)
            self.ub = np.array(func.upper, dtype=float)
        except AttributeError:
            self.lb = np.array(func.bounds.lb, dtype=float)
            self.ub = np.array(func.bounds.ub, dtype=float)

        evaluations_used = 0

        # Initialize population uniformly
        self.pop = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.dim)
        self.fitness = np.empty(self.pop_size)

        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = func(self.pop[i])
            evaluations_used += 1
            if evaluations_used >= self.budget:
                # Budget exhausted during initialization – return best found so far.
                best_idx = np.argmin(self.fitness[:i+1])
                return self.pop[best_idx].copy(), self.fitness[best_idx]

        best_idx = np.argmin(self.fitness)
        best_x = self.pop[best_idx].copy()
        best_y = self.fitness[best_idx]

        # Main DE loop
        for generation in range(self.generations):
            # Dither: F varies per generation
            F = self.F_low + np.random.rand() * (self.F_high - self.F_low)

            for i in range(self.pop_size):
                # Choose three distinct random indices different from i
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutation: DE/rand/1
                mutant = self.pop[r1] + F * (self.pop[r2] - self.pop[r3])
                mutant = self._bounce_back(mutant)

                # Crossover: exponential (geometric segment length)
                # Start at a random dimension, then copy from mutant for a geometrically-distributed
                # number of consecutive dimensions.
                trial = self.pop[i].copy()
                L = np.random.geometric(p=1.0 - self.CR)
                start_dim = np.random.randint(0, self.dim)
                for k in range(L):
                    idx = (start_dim + k) % self.dim
                    trial[idx] = mutant[idx]

                # Ensure boundary after crossover (mutant was already bounced, but trial has some
                # parent dimensions that might now be out of bounds if mutant was reflected? no,
                # only mutant values are copied, which are in bounds. So bounce-back is redundant
                # but safe.)
                trial = self._bounce_back(trial)

                # Evaluate trial
                trial_fitness = func(trial)
                evaluations_used += 1

                # Greedy selection
                if trial_fitness < self.fitness[i]:
                    self.pop[i] = trial
                    self.fitness[i] = trial_fitness
                    # Update global best
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

                # Budget check after each evaluation
                if evaluations_used >= self.budget:
                    return best_x, best_y

        return best_x, best_y
