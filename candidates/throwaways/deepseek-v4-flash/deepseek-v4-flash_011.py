import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE/rand/1/bin) with fixed control parameters F=0.5, CR=0.9.
# Population size is scaled based on dimension and available budget. If the budget is too small
# for even one complete population generation, a simple random search is used instead.
# Search state: A population of candidate solutions (real vectors) stored in a 2D NumPy array.
# Also tracked are the individual objective values and the current best solution found so far.
# Candidate generation: For each target vector three mutually distinct random population members
# (different from the target) are selected. A mutant vector is formed as base + F * (diff1 - diff2).
# Binomial crossover combines the mutant with the target with probability CR; at least one
# dimension is always taken from the mutant.
# Selection and replacement: Greedy selection – the trial vector replaces the target if its
# objective value is lower (better) in a minimization sense.
# Adaptation: None; F and CR are constant. This keeps the code compact and robust across many
# problems, although adaptivity might improve performance on some landscapes.
# Exploration mechanisms: The differential mutation step creates exploratory moves whose
# magnitude is automatically related to the population’s current spread. Crossover further
# mixes components.
# Exploitation mechanisms: The greedy replacement and the maintenance of the best solution
# focus resources on promising regions. The population as a whole converges over time.
# Boundary handling: All candidate vectors are clipped to the decision variable bounds.
# Budget strategy: The population size is chosen as max(4, min(10*dim, budget//5)). If the
# budget is less than this size we simply sample budget uniformly random points. Otherwise we
# perform one initialization evaluation per individual and then as many full generations as
# the remaining budget permits ( budget = popsize + generations * popsize ).
# Closest known influences: Classic DE (Storn & Price, 1995).
# Novelty or unusual aspects: None – a standard implementation with minimal frills.
# Failure modes: Fixed parameters may lead to slow convergence on some problems or premature
# convergence on highly multimodal landscapes. Very low budgets may force the algorithm to
# behave like pure random search, which may be suboptimal if the optimum is not near the
# distribution’s center.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    """Black-box minimisation benchmark algorithm (Differential Evolution)."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # -------------------------------
        # 1. Read bounds from func object
        # -------------------------------
        try:
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.array(func.bounds.lb, dtype=float)
                ub = np.array(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Cannot read bounds from func object")

        dim = self.dim
        budget = self.budget

        # Ensure bounds are 1D arrays of correct length
        if lb.ndim == 0:
            lb = np.full(dim, lb.item())
            ub = np.full(dim, ub.item())
        else:
            lb = lb.ravel()
            ub = ub.ravel()

        # ----------------------------------------------------
        # 2. Determine population size and number of generations
        # ----------------------------------------------------
        # A minimal reasonable population (at least 4) scales with dimension but is capped
        # to consume no more than 1/5 of the total budget for initialization.
        popsize = max(4, min(10 * dim, budget // 5))

        # Budget too small for even one generation -> fall back to random search
        if budget < popsize:
            best_x = None
            best_y = float("inf")
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                y = func(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # Generations: we already spent popsize evaluations on initialization,
        # remaining budget allows floor((budget - popsize) / popsize) full generations.
        generations = (budget - popsize) // popsize
        total_evals = popsize + generations * popsize  # <= budget by construction

        # ---------------------------------------------
        # 3. Initialise population uniformly in bounds
        # ---------------------------------------------
        pop = np.random.uniform(lb, ub, size=(popsize, dim))
        fit = np.array([func(pop[i]) for i in range(popsize)])

        # Track global best
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # Parameters
        F = 0.5
        CR = 0.9

        # --------------------------------
        # 4. Main DE loop over generations
        # --------------------------------
        for _ in range(generations):
            for i in range(popsize):
                # choose three distinct random indices != i
                candidates = list(range(popsize))
                candidates.remove(i)
                r = np.random.choice(candidates, size=3, replace=False)
                a, b, c = r[0], r[1], r[2]

                # Mutation: DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(
                    np.random.rand(dim) < CR,
                    mutant,
                    pop[i]
                )
                # Ensure at least one component from mutant
                trial[j_rand] = mutant[j_rand]

                # Boundary handling: clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_fit = func(trial)

                # Selection (greedy for minimization)
                if trial_fit < fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

        return best_x, best_y
