import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE/rand/1/bin) with fixed parameters for
# black-box minimization. A population of candidate solutions evolves over
# generations using mutation, crossover, and greedy selection.
# Search state: A population of size p stored as a (p,dim) array of real-valued
# vectors, initially sampled uniformly within the search bounds.
# Candidate generation: For each target vector i, three distinct random indices
# a,b,c are chosen (all different from i). The mutant is v = pop[a] + F*(pop[b] - pop[c]).
# The trial vector is formed by binomial crossover with probability Cr.
# Selection and replacement: The trial replaces the target if it yields a lower
# objective value (minimization). The best solution seen so far is tracked.
# Adaptation: None; F = 0.8 and Cr = 0.9 are fixed constants.
# Exploration mechanisms: Mutation via scaled differences between random
# population members introduces diversity; binomial crossover also shuffles
# coordinates across the search space.
# Exploitation mechanisms: Greedy selection gradually replaces worse solutions
# with better ones, focusing the population around promising regions.
# Boundary handling: All coordinates are clamped to the lower and upper bounds
# after mutation and before evaluation.
# Budget strategy: The population size is chosen as max(5, min(4*dim,
# budget//2)) to allow at least a few generations while respecting the budget.
# The main loop runs whole generations until the remaining evaluations are
# fewer than the population size. These leftover evaluations are not used,
# which is a minor inefficiency on very small budgets.
# Closest known influences: Classic DE/rand/1/bin as described by Storn & Price.
# Novelty or unusual aspects: A deliberately simple, no‑frills implementation
# focusing on readability and robustness.
# Failure modes: Fixed parameters may lead to premature convergence on
# multimodal landscapes. On extremely low budgets (< pop_size), the initial
# population is reduced accordingly, but the algorithm may then lack diversity.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ------------------------------------------------------------------
        # 1. Retrieve boundary information
        # ------------------------------------------------------------------
        try:
            lb = np.asarray(func.lower, dtype=np.float64)
            ub = np.asarray(func.upper, dtype=np.float64)
        except AttributeError:
            try:
                lb = np.asarray(func.bounds.lb, dtype=np.float64)
                ub = np.asarray(func.bounds.ub, dtype=np.float64)
            except AttributeError:
                raise ValueError("Cannot locate bounds from func")

        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)

        bounds_span = ub - lb

        # ------------------------------------------------------------------
        # 2. Population size – adapted to budget and dimension
        # ------------------------------------------------------------------
        # Minimum 5, at most 4*dim, and never more than half the budget.
        pop_size = int(min(max(5, 4 * self.dim), self.budget // 2))
        if pop_size < 5:
            pop_size = min(self.budget, 5)  # extremely small budget
        pop_size = max(3, pop_size)  # need at least 3 for mutation

        # ------------------------------------------------------------------
        # 3. Fixed DE parameters
        # ------------------------------------------------------------------
        F = 0.8   # scale factor
        Cr = 0.9  # crossover probability

        # ------------------------------------------------------------------
        # 4. Initialisation
        # ------------------------------------------------------------------
        pop = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        y = np.array([func(x) for x in pop])
        evals = pop_size

        # Track best solution
        best_idx = np.argmin(y)
        best_x = pop[best_idx].copy()
        best_y = y[best_idx]

        # ------------------------------------------------------------------
        # 5. Main evolution loop – full generations
        # ------------------------------------------------------------------
        # Number of complete generations we can run without exceeding the budget.
        gen_max = (self.budget - evals) // pop_size

        for _ in range(gen_max):
            for i in range(pop_size):
                # --- Select three distinct indices different from i ---
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                # --- Mutation (DE/rand/1) ---
                mutant = pop[a] + F * (pop[b] - pop[c])

                # --- Boundary clamping ---
                mutant = np.clip(mutant, lb, ub)

                # --- Binomial crossover ---
                trial = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.random() < Cr or j == j_rand:
                        trial[j] = mutant[j]

                # --- Evaluation ---
                trial_y = func(trial)
                evals += 1

                # --- Greedy selection ---
                if trial_y < y[i]:
                    y[i] = trial_y
                    pop[i] = trial
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()

        # ------------------------------------------------------------------
        # 6. Return best found
        # ------------------------------------------------------------------
        return best_x, best_y
