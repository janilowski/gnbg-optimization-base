import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a classic Differential Evolution (DE/rand/1/bin) as a compact,
#   robust black-box minimizer. It adapts population size to the dimension and budget,
#   uses clipped boundary handling, and tracks evaluations to respect the budget.
# Search state: A population of candidate vectors (pop_size x dim) and their objective
#   values are maintained. The best-known solution is stored separately.
# Candidate generation: For each target vector, three distinct random population members
#   are selected to create a mutant (base + F * difference). Binomial crossover combines
#   the mutant and target to produce a trial vector.
# Selection and replacement: The trial vector is evaluated; if its objective is not
#   worse than the target's, it replaces the target (greedy replacement).
# Adaptation: No parameter adaptation is performed; F=0.8 and CR=0.9 are fixed.
#   Population size is set to 10*dim, but capped at budget//2 and minimum 3.
# Exploration mechanisms: High crossover probability (0.9) and the differential mutation
#   encourage diverse exploration, especially in early generations.
# Exploitation mechanisms: As the population converges, difference vectors shrink,
#   naturally focusing search around promising regions. Greedy selection speeds up
#   convergence.
# Boundary handling: Trial vectors are clipped component-wise to the search bounds.
# Budget strategy: The algorithm stops immediately when the evaluation count reaches
#   the budget. It evaluates all initial points, then runs generations until exhaustion.
#   In the last generation, it may stop mid-population.
# Closest known influences: Standard DE/rand/1/bin as described by Storn & Price (1997).
# Novelty or unusual aspects: None. This is a straightforward implementation intended
#   for reliability and readability.
# Failure modes: On very low budgets (<< 10*dim) the population may be too small to
#   explore effectively. The algorithm may also stagnate if F and CR are ill-suited for
#   the problem, but these parameters work well across many functions.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        """
        Prepare the DE optimizer.
        Parameters: budget (max evaluations), dim (problem dimension).
        """
        self.budget = budget
        self.dim = dim

        # DE control parameters (fixed)
        self.F = 0.8          # mutation factor
        self.CR = 0.9         # crossover rate

        # Population size: scale with dimension but respect budget and minimum of 3.
        self.pop_size = max(3, min(10 * dim, budget // 2))

    def __call__(self, func):
        """
        Run optimization on func (minimization).
        Returns (best_x, best_y) within the evaluation budget.
        func exposes lower/upper or bounds.lb/bounds.ub as 1-D arrays.
        """
        # Read bounds (handle both attribute conventions)
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("func must have .lower/.upper or .bounds.lb/.bounds.ub")

        # Ensure lb, ub are 1-D with correct dimension
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)

        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget

        # Initialize population uniformly in bounds
        pop = np.random.uniform(lb, ub, size=(pop_size, dim))
        f_vals = np.full(pop_size, np.inf)

        # Evaluate initial population, track best
        evals = 0
        best_x = None
        best_y = np.inf

        for i in range(pop_size):
            if evals >= budget:
                break
            y = func(pop[i])
            evals += 1
            f_vals[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Main DE loop
        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break

                # Choose three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutation: base + F * (difference)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])

                # Binomial crossover with target vector i
                trial = np.where(
                    np.random.rand(dim) < self.CR,
                    mutant,
                    pop[i]
                )

                # Ensure crossover always changes at least one component (optional but common)
                # Not strictly necessary; standard DE does not require it.

                # Clip trial to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_y = func(trial)
                evals += 1

                # Selection (greedy)
                if trial_y <= f_vals[i]:
                    pop[i] = trial
                    f_vals[i] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()

        return best_x, best_y
