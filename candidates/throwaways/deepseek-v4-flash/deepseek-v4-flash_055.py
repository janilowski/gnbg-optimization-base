import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a steady-state Differential Evolution (DE) algorithm
#          for black‑box minimization. It evolves a population of candidate solutions by
#          repeatedly applying mutation and crossover, then greedy selection.
# Search state: Maintains a fixed-size population of real vectors, each with an associated
#               objective value (y). The best found solution (best_x) and its value (best_y)
#               are tracked globally.
# Candidate generation: For each mutation step, a mutant vector is created by adding the
#                       scaled difference of two distinct random population members to a
#                       third (DE/rand/1). The scaling factor F is drawn uniformly from
#                       [0.5, 1.0) per mutation to provide adaptive exploration.
# Selection and replacement: After binomial crossover with the current target vector, the
#                            resulting trial vector is evaluated. If its objective value is
#                            lower (better), it replaces the target in the population and
#                            updates the global best if needed.
# Adaptation: The scaling factor F adapts automatically because it is sampled fresh each
#             generation from a wider range, allowing both exploratory and exploitative
#             mutations. The crossover rate CR is static (0.9) and works well across many
#             problems.
# Exploration mechanisms: Large F values (close to 1.0) and moderate crossover rates (0.9)
#                         produce diverse trial vectors. Differential mutation uses the
#                         population’s own geometry to explore the search space.
# Exploitation mechanisms: As the population converges, the difference vectors shrink,
#                          reducing step sizes naturally. Greedy replacement ensures that
#                          only improving solutions survive, focusing the population near
#                          promising basins.
# Boundary handling: If a component of the trial vector falls outside the domain, it is
#                    reflected back symmetrically (midpoint bounce-back) to stay feasible.
# Budget strategy: The algorithm runs a loop that generates exactly one trial vector per
#                  iteration and evaluates it only if the evaluation budget has not been
#                  exhausted. The loop terminates immediately when the budget is used up.
# Closest known influences: Classic DE/rand/1/bin with a dithering F factor (randomly
#                           varied per mutation). Similar to the algorithm described in
#                           Storn & Price (1997).
# Novelty or unusual aspects: None; this is a canonical steady‑state DE variant chosen for
#                             its simplicity, robustness, and zero parameter tuning aside
#                             from population size.
# Failure modes: On separable or highly multimodal functions with very small budgets
#                (< 10×dim) the population may not have enough time to converge. For
#                extremely high dimensions (dim > 100) the fixed crossover rate 0.9 may
#                cause premature convergence; adaptive CR would help.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    """Steady-state Differential Evolution for black-box minimization."""

    def __init__(self, budget: int, dim: int):
        """
        Args:
            budget: Maximum number of objective evaluations allowed.
            dim:    Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        # Population size: at least 10, at most 50, scales moderately with dimension
        self.pop_size = max(10, min(50, 5 * dim))

    def __call__(self, func):
        """Run the optimizer on the given objective function.

        Args:
            func: Callable that accepts a 1-D numpy array and returns a scalar.
                  Must expose bounds via either func.lower / func.upper or
                  func.bounds.lb / func.bounds.ub (both float arrays of length dim).

        Returns:
            (best_x, best_y): Best found solution (1-D numpy array) and its value.
        """
        # --- Read bounds -------------------------------------------------
        try:
            lower = np.asarray(func.lower, dtype=np.float64)
            upper = np.asarray(func.upper, dtype=np.float64)
        except AttributeError:
            try:
                lower = np.asarray(func.bounds.lb, dtype=np.float64)
                upper = np.asarray(func.bounds.ub, dtype=np.float64)
            except AttributeError:
                raise AttributeError(
                    "Cannot read bounds: expected func.lower/upper or func.bounds.lb/ub."
                )
        # Ensure 1-D arrays
        lower = lower.ravel()
        upper = upper.ravel()
        if len(lower) != self.dim or len(upper) != self.dim:
            raise ValueError("Bound arrays must match dimension")

        # --- Initialization ---------------------------------------------
        # Generate a uniform random population
        pop = np.random.uniform(low=lower, high=upper, size=(self.pop_size, self.dim))
        # Evaluate all initial points
        evals = 0
        fx = np.empty(self.pop_size)
        for i in range(self.pop_size):
            fx[i] = func(pop[i])
            evals += 1
            if evals >= self.budget:
                # Budget exhausted during initialization – return best so far
                best_idx = np.argmin(fx[:i+1])
                best_x = pop[best_idx].copy()
                best_y = fx[best_idx]
                return best_x, best_y

        best_idx = np.argmin(fx)
        best_x = pop[best_idx].copy()
        best_y = fx[best_idx]

        # --- DE parameters ----------------------------------------------
        CR = 0.9  # crossover rate
        # F will be sampled per mutation uniformly from [0.5, 1.0)

        # --- Main loop (steady‑state) ----------------------------------
        while evals < self.budget:
            # Choose a random target index
            target_idx = np.random.randint(self.pop_size)

            # Choose three distinct random indices different from target_idx
            indices = list(range(self.pop_size))
            indices.remove(target_idx)
            r1, r2, r3 = np.random.choice(indices, size=3, replace=False)

            # Mutation: DE/rand/1 with dithering F
            F = np.random.uniform(0.5, 1.0)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])

            # Binomial crossover
            j_rand = np.random.randint(self.dim)
            trial = np.empty(self.dim)
            for j in range(self.dim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[target_idx, j]

            # Boundary handling (midpoint reflection)
            # If out of bounds, reflect symmetrically around the bound
            for j in range(self.dim):
                if trial[j] < lower[j]:
                    trial[j] = lower[j] + (lower[j] - trial[j])
                elif trial[j] > upper[j]:
                    trial[j] = upper[j] - (trial[j] - upper[j])
                # Clip to bound if reflection still fails (rare due to numeric)
                trial[j] = np.clip(trial[j], lower[j], upper[j])

            # Evaluate trial
            fx_trial = func(trial)
            evals += 1

            # Greedy selection
            if fx_trial < fx[target_idx]:
                pop[target_idx] = trial
                fx[target_idx] = fx_trial
                # Update global best
                if fx_trial < best_y:
                    best_x = trial.copy()
                    best_y = fx_trial

            # Check budget again (loop condition will also stop)
            if evals >= self.budget:
                break

        return best_x, best_y
