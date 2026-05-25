import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE/rand/1/bin) with simple clipping boundary handling.
# Search state: A fixed‑size population of candidate vectors; the best solution ever found is tracked.
# Candidate generation: For each target vector, three distinct random vectors are selected,
#   a mutant is created as v1 + F * (v2 - v3), then binomial crossover with the target produces a trial vector.
# Selection and replacement: Greedy – the trial replaces the target if it yields a lower function value.
# Adaptation: No self‑adaptation; parameters F = 0.8 and CR = 0.9 are static.
# Exploration: Mutation with scaled difference vectors introduces diversity; binomial crossover shuffles gene information.
# Exploitation: Greedy selection drives the population toward better regions; convergence of the population increases local search.
# Boundary handling: Each coordinate of the trial vector is clipped (truncated) to the search interval [lower, upper].
# Budget strategy: The initial population uses pop_size evaluations; remaining budget is split into full generations
#   and one partial generation to use exactly the given budget.
# Closest known influences: Classic differential evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None; a straightforward, textbook implementation.
# Failure modes: Fixed parameters may not suit all landscapes; may stagnate in multimodal or deceptive functions;
#   scaling may be poor in very high dimensions (no adaptation of F/CR).
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ---------- bounds ----------
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            bounds = func.bounds
            lb = np.asarray(bounds.lb, dtype=float)
            ub = np.asarray(bounds.ub, dtype=float)
        else:
            raise ValueError("Cannot find objective bounds")

        if lb.ndim == 0:                     # scalar bound repeated for each dimension
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)

        # ---------- algorithm parameters ----------
        pop_size = min(self.budget, max(10, 4 * self.dim))
        pop_size = max(pop_size, 4)           # DE needs at least 4 individuals
        F = 0.8
        CR = 0.9

        # ---------- initialisation ----------
        pop = np.random.uniform(lb, ub, size=(pop_size, self.dim))
        fit = np.array([func(x) for x in pop])
        evals = pop_size

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # ---------- remaining budget ----------
        remaining = self.budget - evals
        full_gens = remaining // pop_size
        leftover = remaining % pop_size

        # ---------- helper: one DE iteration for a given index ----------
        def de_step(tgt_idx):
            nonlocal evals, best_x, best_y
            # choose three distinct random indices (different from tgt_idx)
            indices = [j for j in range(pop_size) if j != tgt_idx]
            r = np.random.choice(indices, size=3, replace=False)
            a, b, c = r[0], r[1], r[2]
            # mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            # binomial crossover
            cross = np.random.rand(self.dim) < CR
            if not cross.any():
                cross[np.random.randint(self.dim)] = True
            trial = np.where(cross, mutant, pop[tgt_idx])
            # clip to bounds
            trial = np.clip(trial, lb, ub)
            # evaluation
            trial_fit = func(trial)
            evals += 1
            # greedy selection
            if trial_fit < fit[tgt_idx]:
                pop[tgt_idx] = trial
                fit[tgt_idx] = trial_fit
                if trial_fit < best_y:
                    best_y = trial_fit
                    best_x = trial.copy()

        # ---------- main generations ----------
        for _ in range(full_gens):
            for i in range(pop_size):
                de_step(i)

        # ---------- leftover evaluations (partial generation) ----------
        for i in range(leftover):
            de_step(i)

        return best_x, best_y
