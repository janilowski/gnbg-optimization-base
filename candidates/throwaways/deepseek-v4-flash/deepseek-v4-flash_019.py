import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE/rand/1/bin) minimizer that
# works with the GNBG black-box benchmark. It initializes a population of
# candidate solutions uniformly in the search domain, then iterates a
# mutation-crossover-selection loop until the evaluation budget is exhausted.
# The best found solution and its objective value are returned.
# Search state: A matrix of shape (pop_size, dim) storing the current
# population, plus an array of corresponding objective values.
# Candidate generation: For each target vector, a mutant is built by adding
# the scaled difference of two random population vectors to a third random
# vector (DE/rand/1). The trial vector is formed via binomial crossover with
# the target vector.
# Selection and replacement: The trial vector replaces the target if and only
# if its objective value is better (lower for minimization). This greedy
# selection ensures monotonic improvement in each generation.
# Adaptation: None. The parameters (F=0.8, CR=0.9) are fixed.
# Exploration mechanisms: The difference vector provides random step sizes
# that adapt implicitly to the population’s spread. Crossover mixes
# components from the target and mutant, preserving diversity.
# Exploitation mechanisms: As the population converges, difference vectors
# shrink, focusing the search around the current best region. Greedy
# selection drives the population toward better fitness.
# Boundary handling: All candidate positions are clipped to the feasible
# box defined by func.lower/upper (or func.bounds). This is simple and
# ensures feasibility.
# Budget strategy: The number of objective evaluations is tracked exactly.
# The main loop terminates as soon as the budget (minus the initial
# evaluation cost) would be exceeded. A final best candidate is selected
# from all evaluated points.
# Closest known influences: Classic DE/rand/1/bin (Storn & Price, 1997).
# Novelty or unusual aspects: None; this is a textbook implementation.
# Failure modes: May stagnate on highly multimodal or deceptive landscapes,
# especially with a small budget. Fixed parameters may not be optimal for
# all problem dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # -------- read bounds --------
        try:
            lo = np.asarray(func.lower, dtype=float)
            hi = np.asarray(func.upper, dtype=float)
        except AttributeError:
            try:
                lo = np.asarray(func.bounds.lb, dtype=float)
                hi = np.asarray(func.bounds.ub, dtype=float)
            except AttributeError:
                raise ValueError("Cannot find bounds from func")

        # ensure broadcast works
        lo = np.broadcast_to(lo, (self.dim,))
        hi = np.broadcast_to(hi, (self.dim,))

        # -------- population size --------
        # at least 5 individuals, at most budget//2, scaling with dim
        pop_size = max(5, min(self.budget // 2, 10 * self.dim))
        # ensure we have at least 3 for DE mutation
        if pop_size < 3:
            pop_size = 3

        # -------- DE parameters (fixed) --------
        F = 0.8       # mutation scaling
        CR = 0.9      # crossover probability

        # -------- initialisation --------
        pop = lo + np.random.rand(pop_size, self.dim) * (hi - lo)
        fit = np.full(pop_size, np.inf)
        evals = 0

        for i in range(pop_size):
            fit[i] = func(pop[i])
            evals += 1

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # -------- main loop --------
        while evals < self.budget:
            # one generation: generate trial for each target
            # we will produce trials and evaluate them, but we stop
            # if the remaining budget would be exceeded.
            generation_evals = 0
            trials = np.empty_like(pop)
            trial_ok = np.zeros(pop_size, dtype=bool)

            for i in range(pop_size):
                # pick three distinct random indices != i
                choices = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = np.random.choice(choices, 3, replace=False)

                # mutant vector
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # clip to bounds
                mutant = np.clip(mutant, lo, hi)

                # binomial crossover
                # random integer for at least one component from mutant
                j_rand = np.random.randint(self.dim)
                trial = np.where(
                    np.random.rand(self.dim) < CR,
                    mutant,
                    pop[i]
                )
                # ensure at least one component from mutant
                trial[j_rand] = mutant[j_rand]

                trials[i] = np.clip(trial, lo, hi)
                trial_ok[i] = True
                generation_evals += 1

                # if we would exceed budget after generating this trial,
                # stop generating more (do not evaluate yet)
                if evals + generation_evals >= self.budget:
                    # we break the loop, but first we must decide which
                    # trials to evaluate so far. We'll evaluate those that
                    # we have generated (including current) in a second loop.
                    # For simplicity, we mark that we stop after this trial.
                    break

            # evaluate the trials that were generated
            for i in range(pop_size):
                if not trial_ok[i]:
                    continue
                # ensure we have budget left (we may have stopped mid-generation)
                if evals >= self.budget:
                    break
                trial_fit = func(trials[i])
                evals += 1
                if trial_fit < fit[i]:
                    pop[i] = trials[i]
                    fit[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trials[i].copy()

            # if budget exhausted, exit loop
            if evals >= self.budget:
                break

        return best_x, best_y
