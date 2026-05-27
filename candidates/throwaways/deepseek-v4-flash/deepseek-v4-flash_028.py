# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A classic Differential Evolution (DE/rand/1/bin) algorithm for black-box minimization. It evolves a population of candidate solutions by mutating and recombining them, then applying one-to-one greedy selection. The algorithm respects the evaluation budget exactly.
# Search state: A population of `N` vectors of dimension `dim`, each with a fitness value stored alongside. The best solution ever seen is tracked in `best_x` and `best_y`.
# Candidate generation: For each population member, a mutant vector is created as the base vector plus a scaled difference of two other distinct random members. Crossover (binomial) combines the mutant with the current member to produce a trial vector.
# Selection and replacement: After evaluating the trial vector, if its fitness is less than or equal to the current member’s fitness, the current member is replaced. Otherwise it remains unchanged. No archive or crowding is used.
# Adaptation: The parameters are static: crossover rate CR=0.9 and scaling factor F drawn uniformly in [0.5, 1.0] per mutation (dither) to encourage exploration.
# Exploration mechanisms: Mutation uses randomly selected population members (exploration through difference vectors) and dither increases variability. The population is maintained and diverse except when convergence occurs.
# Exploitation mechanisms: The greedy selection and the fact that good solutions survive focus the search around promising areas.
# Boundary handling: Trial vectors are clipped componentwise to the lower and upper bounds.
# Budget strategy: Population size `N` is computed as `min(10*dim, budget//5)` but at least 4. The number of generations is then `budget // N` (minus one for initialization). The algorithm never exceeds `budget` evaluations because it checks remaining budget before each function call.
# Closest known influences: Standard DE/rand/1/bin as described by Storn and Price (1997).
# Novelty or unusual aspects: Nothing novel; it is a straightforward implementation of a well-known algorithm.
# Failure modes: May converge prematurely on noisy or highly multimodal landscapes; fixed F and CR may not suit all problems; poor performance on separable or highly conditioned functions without rotation.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Differential Evolution optimizer for black-box minimization."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ------------------------------------------------------------------
        # 1. Extract bounds from the problem object
        # ------------------------------------------------------------------
        # Support both func.lower/upper and func.bounds interface.
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            # GNBG benchmark uses func.bounds.lb / .ub
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective does not provide lower/upper bounds.")

        # Ensure 1-D arrays of the correct dimension
        lb = np.broadcast_to(lb, self.dim).copy()
        ub = np.broadcast_to(ub, self.dim).copy()

        # ------------------------------------------------------------------
        # 2. Determine population size and generation count
        # ------------------------------------------------------------------
        budget = self.budget
        if budget < 4:
            raise ValueError("Budget must be at least 4 for DE population.")

        # Population size: between 4 and 10*dim, scaled so that at least
        # one generation (after initialization) can be run.
        N = min(max(4, 10 * self.dim), budget // 2)   # at least 2 generations
        # Actually ensure we have at least one generation beyond initialization
        N = min(N, budget - 1)   # leave at least 1 evaluation for generation
        # Recompute generations that fit exactly
        generations = (budget - N) // N   # after initial eval, budget//N full generations
        # But if budget is small, generations may be 0; then just do initial eval.
        if generations < 0:
            generations = 0

        # ------------------------------------------------------------------
        # 3. Initialize population uniformly in bounds
        # ------------------------------------------------------------------
        pop = np.random.uniform(lb, ub, size=(N, self.dim))
        fitness = np.full(N, np.inf)
        best_x = None
        best_y = np.inf
        remaining = budget   # count of allowed function evaluations

        # Helper to evaluate and track best
        def evaluate(x):
            nonlocal remaining, best_x, best_y
            if remaining <= 0:
                raise RuntimeError("Budget exhausted – this should not happen.")
            y = func(x)
            remaining -= 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return np.float64(y)

        # Initial evaluation of all population members
        for i in range(N):
            fitness[i] = evaluate(pop[i])
            if remaining == 0:
                # Budget exhausted – cannot continue; return best so far
                return best_x, best_y

        # ------------------------------------------------------------------
        # 4. Main DE loop (rand/1/bin)
        # ------------------------------------------------------------------
        CR = 0.9          # crossover rate

        for _ in range(generations):
            # Process each target vector
            for i in range(N):
                # Check if any evaluations remain
                if remaining == 0:
                    return best_x, best_y

                # Choose three distinct random indices different from i
                candidates = list(range(N))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                # Mutation (dither F) – scaling factor varies per mutant
                F = np.random.uniform(0.5, 1.0)
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Boundary clipping
                mutant = np.clip(mutant, lb, ub)

                # Binomial crossover
                # Pre‑allocate trial = copy of current
                trial = pop[i].copy()
                # Select a random index that will always be taken from mutant
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluate trial
                trial_fitness = evaluate(trial)
                if remaining == 0:
                    return best_x, best_y

                # One‑to‑one greedy selection
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

        # Return best encountered over the whole run
        return best_x, best_y
