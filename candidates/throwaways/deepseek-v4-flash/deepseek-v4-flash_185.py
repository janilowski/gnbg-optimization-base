import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE) with the classic rand/1/bin strategy.
#          Fixed control parameters (F=0.8, CR=0.9). Population size is chosen
#          adaptively based on dimension and budget, capped at 100.
# Search state: A population of `pop_size` candidate solutions (real vectors)
#               uniformly distributed in the search space at initialization.
# Candidate generation: For each target individual, three distinct random population
#                       members (excluding the target) are selected to form a
#                       difference vector, which is scaled by F and added to a
#                       base vector (the first random individual). The resulting
#                       mutant is crossed with the target using binomial crossover
#                       with probability CR.
# Selection and replacement: Greedy one-to-one – if the trial vector has a lower
#                            (or equal) objective value than the target, it replaces
#                            the target in the population.
# Adaptation: None; F and CR remain constant throughout the run.
# Exploration mechanisms: The scaled difference vector provides random perturbations
#                         that can jump across the search space; crossover creates
#                         mixed offspring.
# Exploitation mechanisms: Greedy replacement gradually improves the population;
#                          the best solution is tracked and returned.
# Boundary handling: Points outside the box are clamped to the nearest bound.
# Budget strategy: Evaluations are counted one by one. An initial population is
#                  evaluated (pop_size evals). Then, as long as budget remains,
#                  the algorithm proceeds into generations: for each target
#                  individual a trial is generated and evaluated, consuming one
#                  evaluation per trial. The loop stops exactly when no further
#                  evaluation fits in the remaining budget.
# Closest known influences: Standard Differential Evolution (Price, Storn, Lampinen).
# Novelty or unusual aspects: None – a straightforward implementation of DE.
# Failure modes: On very low budgets (< pop_size) only random sampling is performed;
#                high-dimensional problems may require more generations than the
#                budget permits, leading to poor convergence.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """
    Differential Evolution minimizer for the GNBG black-box benchmark.
    """

    def __init__(self, budget: int, dim: int):
        """
        Args:
            budget: Maximum number of function evaluations allowed.
            dim:    Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

        # Differential Evolution control parameters (fixed)
        self.F = 0.8          # mutation scaling factor
        self.CR = 0.9         # crossover probability

        # Population size: tune for dimension, but not too large
        # Prefer at least 10 individuals, at most 100.
        # Ensure we can run at least a few generations.
        self.pop_size = min(100, max(10, 5 * dim))
        # If budget is extremely small, reduce population size to at least 1
        if self.budget < self.pop_size:
            self.pop_size = self.budget

        # Bounds will be read from func during __call__
        self.lb = None
        self.ub = None

    def __call__(self, func):
        """
        Run the minimizer.

        Args:
            func: The objective function to minimize. Provides bounds via
                  func.lower / func.upper or func.bounds.lb / func.bounds.ub.
        Returns:
            (best_x, best_y) where best_x is a 1-D numpy array and best_y is the
            minimum objective value found.
        """
        # --- Read bounds ---
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        else:
            # Fallback to sensible defaults if bounds are missing (should not happen)
            lb = np.full(self.dim, -100.0)
            ub = np.full(self.dim, 100.0)

        self.lb = lb
        self.ub = ub

        # --- Initialize population uniformly in the box ---
        rng = np.random.default_rng()  # relies on global numpy seed
        pop = rng.uniform(low=lb, high=ub, size=(self.pop_size, self.dim))

        # Evaluate initial population
        fitness = np.full(self.pop_size, np.inf)
        evals_used = 0
        best_x = None
        best_y = np.inf

        for i in range(self.pop_size):
            if evals_used >= self.budget:
                break
            y = func(pop[i])
            fitness[i] = y
            evals_used += 1
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # --- Main DE loop ---
        # We keep generating generations as long as we have at least one evaluation left.
        while evals_used < self.budget:
            # For each target individual, create a trial and evaluate it
            for i in range(self.pop_size):
                if evals_used >= self.budget:
                    break

                # --- Mutation (rand/1) ---
                # Select three distinct random indices different from i
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                # Shuffle and take first three (this works fine with small pop)
                rng.shuffle(candidates)
                a, b, c = candidates[0], candidates[1], candidates[2]

                # Mutant vector
                mutant = pop[a] + self.F * (pop[b] - pop[c])

                # --- Crossover (binomial) ---
                # Binomial crossover: copy from mutant with prob CR, otherwise from target
                trial = pop[i].copy()
                for j in range(self.dim):
                    if rng.random() < self.CR:
                        trial[j] = mutant[j]
                # Ensure at least one component comes from the mutant (standard DE)
                j_rand = rng.integers(self.dim)
                trial[j_rand] = mutant[j_rand]

                # --- Boundary handling (clamp) ---
                trial = np.clip(trial, self.lb, self.ub)

                # --- Evaluation ---
                f_trial = func(trial)
                evals_used += 1

                # --- Selection (greedy) ---
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()

        # Ensure best_x is always returned (even if budget=0, though that's unlikely)
        if best_x is None and self.pop_size > 0:
            best_x = pop[0].copy()
            best_y = fitness[0]

        return best_x, best_y
