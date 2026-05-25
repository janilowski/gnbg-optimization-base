import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a black-box minimisation algorithm based on
#          Differential Evolution (DE) with the classic rand/1/bin strategy.
#          The population size scales logarithmically with dimension to remain
#          efficient across a wide range of dimensionalities.  For extremely
#          small budgets that cannot support a full initial population, the
#          algorithm falls back to pure random search.
# Search state: The algorithm maintains a population of candidate solutions
#               (real vectors) and their corresponding objective values.
# Candidate generation: Each generation uses differential mutation:
#                       mutant = base + F * (diff1 - diff2) where base, diff1, diff2
#                       are three mutually distinct random population members.
#                       Then binomial crossover combines the mutant with the
#                       current target vector.
# Selection and replacement: Greedy one‑to‑one selection: a new trial replaces
#                            its parent if it yields a lower (better) objective value.
# Adaptation: No parameter adaptation is used; F and CR are fixed.
# Exploration mechanisms: Mutation uses randomly chosen population members,
#                         which promotes diversity.  The crossover rate CR
#                         allows mixing components from the mutant and the parent.
# Exploitation mechanisms: Greedy selection pushes the population towards
#                          low‑objective regions.  The best solution ever found
#                          is explicitly stored.
# Boundary handling: Any coordinate that falls outside the feasible box is
#                    clipped to the nearest bound.
# Budget strategy: The algorithm runs whole generations (each generation consumes
#                  population‑size evaluations) as long as the remaining budget
#                  is sufficient.  If the initial budget is too small even for a
#                  single population, a pure random search is used instead.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: The population size formula
#                             max(5, int(4 + 3 * log(dim))) is chosen to keep
#                             the algorithm lightweight for high dimensions.
# Failure modes: The algorithm may stagnate in multimodal landscapes because
#                it uses no adaptation, diversity preservation, or restart.
#                Very low budgets may lead to poor results.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """A simple Differential Evolution minimizer."""
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # DE parameters (fixed)
        self.F = 0.8          # mutation factor
        self.CR = 0.9         # crossover probability
        # Population size: grows slowly with dimension to stay efficient
        self.popsize = max(5, int(4 + 3 * np.log(dim)))

    def _get_bounds(self, func):
        """Return lower and upper bound arrays (numpy vectors)."""
        # Try standard attributes
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)
        else:
            # Fallback: assume unit hypercube (should not happen in GNBG)
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)
        return lb, ub

    def __call__(self, func):
        lb, ub = self._get_bounds(func)
        dim = self.dim
        popsize = self.popsize
        budget = self.budget

        # Fallback: if budget is too small for a proper DE run, use random search
        if budget < popsize + 1:
            best_x = None
            best_y = float('inf')
            for _ in range(budget):
                x = lb + np.random.uniform(0, 1, dim) * (ub - lb)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # Initialisation: create population uniformly in [lb, ub]
        pop = lb + np.random.uniform(0, 1, (popsize, dim)) * (ub - lb)
        # Evaluate initial population
        pop_y = np.array([func(p) for p in pop])
        evals = popsize

        # Track the best solution
        best_idx = np.argmin(pop_y)
        best_x = pop[best_idx].copy()
        best_y = pop_y[best_idx]

        # Main DE loop – run as many whole generations as budget allows
        while evals + popsize <= budget:
            for i in range(popsize):
                # Choose three distinct random indices different from i
                candidates = list(range(popsize))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                # Mutation: DE/rand/1
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                # Crossover: binomial
                jrand = np.random.randint(dim)
                trial = np.array([mutant[j] if (np.random.rand() < self.CR or j == jrand)
                                  else pop[i, j] for j in range(dim)])
                # Boundary clipping
                trial = np.clip(trial, lb, ub)

                # Evaluation
                trial_y = func(trial)
                evals += 1

                # Selection
                if trial_y < pop_y[i]:
                    pop[i] = trial
                    pop_y[i] = trial_y
                    # Update global best
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()

        return best_x, best_y
