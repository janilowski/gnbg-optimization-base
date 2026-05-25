# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A population‑based stochastic optimizer that evolves a set of candidate solutions using differential evolution (DE). Each iteration creates mutant vectors by combining three randomly chosen individuals, applies binomial crossover to produce trial vectors, evaluates them, and replaces the target when the trial improves the objective. Mutation scaling (F) and crossover probability (CR) are re‑drawn occasionally to adapt the search behavior during the run.

# Search state: The algorithm maintains a population of `pop_size` vectors that collectively represent the current set of candidate solutions. It also tracks the best solution found so far (best_x, best_y) and the number of function evaluations consumed (evals).

# Candidate generation: For each individual i, three distinct individuals (r0, r1, r2) are selected at random, excluding i. A mutant vector is built as `mutant = pop[i] + F_i * (pop[r0] - pop[r1])`. The scaling factor F_i is either a fixed value (0.5) or drawn uniformly in [0.1, 1.0] with a small probability to allow exploration.

# Selection and replacement: A trial vector is created by binomial crossover: for each dimension j, copy the mutant value with probability CR_i (else keep the original target value). The trial is evaluated on the true objective (minimization). If the trial’s fitness is not worse than the target’s fitness, the target is replaced by the trial; otherwise the target is retained. The global best solution is updated whenever a better trial appears.

# Adaptation: Both F and CR are occasionally re‑sampled per individual, emulating the jitter used in the JDE algorithm. This lets the algorithm self‑adapt the balance between exploration (large F) and exploitation (small F) without external control.

# Exploration mechanisms: Large or randomly drawn F values generate diverse mutants; a moderate CR (default 0.9) spreads information from the mutant across many dimensions, fostering broad exploration.

# Exploitation mechanisms: Small F values focus perturbations around the current individual, enabling fine‑grained refinement near promising regions.

# Boundary handling: All candidate vectors are clipped component‑wise to the problem’s lower and upper bounds after mutation and crossover to keep the search within the feasible space.

# Budget strategy: The algorithm never performs more than the supplied evaluation `budget`. It evaluates the initial population, then runs DE generations until the budget is exhausted, breaking early if needed. The best solution discovered at any point is returned.

# Closest known influences: Classic differential evolution (Storn & Price, 1997) and the self‑adaptive JDE variant (Brest et al., 2006). The implementation mirrors the (μ+λ) selection style, where each individual competes with its own trial.

# Novelty or unusual aspects: Using per‑individual jitter for both F and CR within a single loop keeps the code compact while still providing adaptive control. No external libraries are used beyond NumPy for random sampling and vector operations.

# Failure modes: When the evaluation budget is far smaller than the dimensionality times the recommended population size, the algorithm may not converge to a good solution. Very high dimensional problems may require a larger population than the heuristic `4*dim` to maintain diversity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    A compact differential‑evolution optimizer for black‑box minimization.
    Implements the required interface: __init__(budget, dim) and __call__(func).
    """

    def __init__(self, budget, dim):
        """
        Parameters
        ----------
        budget : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimensionality of the decision space.
        """
        self.budget = int(budget)
        self.dim = int(dim)
        # Heuristic population size: enough to cover the space but not excessive.
        self.pop_size = max(10, 4 * dim)

    def __call__(self, func):
        """
        Run the optimizer on the given function.

        Parameters
        ----------
        func : callable
            Black‑box objective to minimize. It must accept a 1‑D NumPy array
            of length `dim` and return a scalar.

        Returns
        -------
        best_x : np.ndarray
            Best solution found (decision vector).
        best_y : float
            Corresponding objective value.
        """
        # ------------------------------------------------------------------
        # 1. Determine search bounds (lower, upper) from the function object.
        # ------------------------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # Assume func.bounds is a namespace with .lb and .ub attributes.
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: unbounded search – use huge interval.
            lower = np.full(self.dim, -1e308, dtype=float)
            upper = np.full(self.dim, 1e308, dtype=float)

        # ------------------------------------------------------------------
        # 2. Initialise random population within the bounds.
        # ------------------------------------------------------------------
        pop = np.random.uniform(lower, upper, size=(self.pop_size, self.dim))
        fitness = np.empty(self.pop_size, dtype=float)

        # ------------------------------------------------------------------
        # 3. Evaluate initial population respecting the budget.
        # ------------------------------------------------------------------
        evals = 0
        max_init = min(self.pop_size, self.budget)
        for i in range(max_init):
            fitness[i] = func(pop[i])
            evals += 1

        # If the budget is exhausted already, return the best of the evaluated.
        if evals >= self.budget:
            best_idx = np.argmin(fitness[:evals])
            return pop[best_idx].copy(), float(fitness[best_idx])

        # Track global best.
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = float(fitness[best_idx])

        # ------------------------------------------------------------------
        # 4. Main DE loop – generate, evaluate, select, adapt.
        # ------------------------------------------------------------------
        # Default control parameters.
        F_default = 0.5
        CR_default = 0.9

        while evals < self.budget:
            for i in range(self.pop_size):
                # ----- Mutation -----
                # Select three distinct indices different from i.
                indices = list(range(self.pop_size))
                indices.remove(i)
                r0, r1, r2 = np.random.choice(indices, 3, replace=False)

                # JDE‑style jitter: occasionally re‑draw F and CR.
                if np.random.rand() < 0.1:
                    F_i = np.random.uniform(0.1, 1.0)
                else:
                    F_i = F_default

                mutant = pop[i] + F_i * (pop[r0] - pop[r1])

                # ----- Crossover -----
                if np.random.rand() < 0.1:
                    CR_i = np.random.uniform(0.0, 1.0)
                else:
                    CR_i = CR_default

                # Binomial crossover: copy each dimension from mutant with probability CR_i.
                mask = np.random.rand(self.dim) < CR_i
                trial = np.where(mask, mutant, pop[i])

                # ----- Repair bounds -----
                trial = np.clip(trial, lower, upper)

                # ----- Evaluate trial -----
                if evals >= self.budget:
                    return best_x, best_y

                y_trial = func(trial)
                evals += 1

                # ----- Selection (μ + λ style, μ = pop_size) -----
                if y_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = y_trial
                    if y_trial <= best_y:
                        best_x = trial.copy()
                        best_y = float(y_trial)

        # ------------------------------------------------------------------
        # 5. Return best found.
        # ------------------------------------------------------------------
        return best_x, best_y
