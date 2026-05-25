import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a classic Differential Evolution (DE/rand/1/bin) as a black‑box minimizer.
#          The method is compact, relies only on numpy, and is dimension‑agnostic.
# Search state: A population of candidate solutions stored as a 2D array (pop_size x dim).
#               The scalar fitness of each candidate is kept in a separate array.
# Candidate generation: For each target vector in the population, a mutant is formed by adding
#                       a scaled difference of two distinct random population members to a third.
#                       A binomial crossover combines the mutant with the target to produce a trial.
# Selection and replacement: The trial replaces the target if its fitness is better (lower).
# Exploration mechanisms: The differential mutation uses random indices; crossover probability CR
#                         controls gene mixing. The scaling factor F influences step size.
# Exploitation mechanisms: Population gradually converges as better solutions replace worse ones.
#                          The best solution is tracked globally.
# Boundary handling: Mutants and trial vectors are clipped component‑wise to the domain bounds.
# Budget strategy: The population size is set to min(10*dim, max(5, budget//4)), then generations
#                  run until remaining evaluations are exhausted. Each generation consumes exactly
#                  pop_size evaluations (one trial per individual).
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None; a straightforward implementation without any adaptive parameters.
# Failure modes: May converge prematurely with small population or insufficient generations.
#                Very high‑dimensional problems (>50) might need larger pop or tuned F/CR.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        """Initialize the differential evolution algorithm.

        Args:
            budget: Maximum number of function evaluations.
            dim: Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # Algorithm parameters – classic values for DE/rand/1/bin
        self.F = 0.8          # mutation scaling factor
        self.CR = 0.9         # crossover probability

        # Population size: trade‑off between exploration and evaluations per generation.
        # Use at least 5 individuals, at most 10*dim, but also restrict to avoid wasting budget.
        # We want enough generations to allow convergence.
        n_max = min(10 * dim, self.budget // 2)   # at least a few generations
        self.pop_size = max(5, min(n_max, 50 * dim))   # cap at 50*dim (rarely hit)
        # Further enforce that pop_size <= budget (e.g., for very small budget)
        self.pop_size = min(self.pop_size, self.budget)
        # Ensure at least 3 individuals for mutation (need 3 distinct indices)
        if self.pop_size < 3:
            self.pop_size = min(3, self.budget)
        # Number of generations we can run after initialization
        self.gen_max = (self.budget - self.pop_size) // self.pop_size if self.pop_size > 0 else 0

    def __call__(self, func):
        """Run the DE minimizer.

        Args:
            func: The objective function to minimize. It is expected to have
                  .lower and .upper attributes (or .bounds.lb / .bounds.ub) defining the domain.

        Returns:
            (best_x, best_y): The best found solution and its fitness.
        """
        # Read bounds from function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide lower/upper or bounds.lb/bounds.ub")

        # Vectorise for convenience
        lb = np.full(self.dim, lb) if lb.ndim == 0 else lb
        ub = np.full(self.dim, ub) if ub.ndim == 0 else ub

        # Initialise population uniformly in the domain
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))

        # Evaluate initial population
        fitness = np.array([func(x) for x in pop])
        evals = self.pop_size

        # Track global best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Prepare arrays for mutation indices (avoid re‑allocation)
        r1 = np.empty(self.pop_size, dtype=int)
        r2 = np.empty(self.pop_size, dtype=int)
        r3 = np.empty(self.pop_size, dtype=int)

        # Main generation loop
        for gen in range(self.gen_max):
            if evals >= self.budget:
                break

            # Pre‑compute three distinct random indices for each target vector
            for i in range(self.pop_size):
                # Indices must be distinct from each other and from i
                while True:
                    r1[i] = np.random.randint(0, self.pop_size)
                    r2[i] = np.random.randint(0, self.pop_size)
                    r3[i] = np.random.randint(0, self.pop_size)
                    if len({r1[i], r2[i], r3[i], i}) == 4:
                        break

            # For each target vector, create a trial vector
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break  # no more evaluations allowed

                # Mutation: v = r1 + F * (r2 - r3)
                mutant = pop[r1[i]] + self.F * (pop[r2[i]] - pop[r3[i]])

                # Crossover: trial = crossover(mutant, target)
                # Binomial crossover: for each dimension, if rand < CR or j == jrand
                jrand = np.random.randint(0, self.dim)
                trial = np.where(
                    np.random.rand(self.dim) < self.CR,
                    mutant,
                    pop[i]
                )
                # Ensure at least one dimension from mutant (jrand forced)
                trial[jrand] = mutant[jrand]

                # Boundary handling – clip to domain
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_fit = func(trial)
                evals += 1

                # Selection – replace target if trial is better
                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

        return best_x, best_y
