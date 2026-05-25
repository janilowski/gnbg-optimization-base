import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A differential evolution variant (DE/rand/1/bin) with individual-wise adaptive control parameters inspired by jDE.
# Search state: A population of NP candidate solutions, each storing its own F and CR values.
# Candidate generation: For each target individual, a mutant vector is created using three distinct random population members (r0 + F*(r1-r2)). Binomial crossover with the target yields a trial vector.
# Selection and replacement: Greedy selection: the trial replaces the target if its objective value is strictly better (lower). The successful F and CR values are retained; otherwise they are newly sampled on the next attempt.
# Adaptation: Each individual's F and CR are resampled from uniform distributions with small probability (0.1) before each mutation. After a successful replacement, the new parameters are kept; otherwise the old ones are restored.
# Exploration mechanisms: Mutation using random difference vectors and a wide range of F (0.1–0.9) provides exploration. Reflection boundary handling preserves diversity by bouncing out-of-bound coordinates inward.
# Exploitation mechanisms: Crossover with the target and greedy replacement drive convergence. Adaptive parameters allow fine‑tuning of step sizes and linkage.
# Boundary handling: Reflection – coordinates that fall outside [lb, ub] are reflected around the violated boundary (if reflection still fails, clipped to the bound).
# Budget strategy: The population size NP is chosen so that at least half of the budget is left for iterations. Initialisation consumes NP evaluations; each trial uses one evaluation. Total evaluations = NP + iterations <= budget.
# Closest known influences: jDE (Brest et al., 2006) for adaptation; standard DE/rand/1/bin.
# Novelty or unusual aspects: None – a clean, robust implementation of a well‑known algorithm.
# Failure modes: May struggle on highly multimodal landscapes; premature convergence when population size is too small; limited performance on very high‑dimensional problems if the budget is tight.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Minimisation through adaptive Differential Evolution (DE/rand/1/bin, jDE style)."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ---------- Extract bounds ----------
        try:
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        except AttributeError:
            try:
                lb = np.array(func.bounds.lb, dtype=float)
                ub = np.array(func.bounds.ub, dtype=float)
            except AttributeError:
                raise TypeError("Cannot determine bounds from func object")

        dim = self.dim
        budget = self.budget

        # ---------- Population size ----------
        # at least 4, no more than half the budget, and no more than 5*dim
        NP = max(4, min(budget // 2, 5 * dim))
        if budget < 4:                  # very small budget – pure random search
            NP = budget

        # ---------- Initialise population ----------
        pop = np.random.uniform(lb, ub, size=(NP, dim))
        y = np.empty(NP)
        for i in range(NP):
            y[i] = func(pop[i])

        evals_used = NP
        best_idx = np.argmin(y)
        best_x = pop[best_idx].copy()
        best_y = y[best_idx]

        # initialise F and CR for each individual
        F = np.random.uniform(0.1, 0.9, size=NP)
        CR = np.random.uniform(0, 1, size=NP)

        # parameters for adaptation (jDE style)
        tau1 = 0.1
        tau2 = 0.1
        Fl = 0.1
        Fu = 0.9

        # ---------- Main loop ----------
        while evals_used < budget:
            for i in range(NP):
                if evals_used >= budget:
                    break

                # ---- adaptation of F and CR for this individual ----
                F_new = F[i]
                CR_new = CR[i]
                if np.random.rand() < tau1:
                    F_new = Fl + np.random.rand() * (Fu - Fl)
                if np.random.rand() < tau2:
                    CR_new = np.random.rand()

                # ---- choose three distinct indices different from i ----
                candidates = list(range(NP))
                candidates.remove(i)
                r = np.random.choice(candidates, size=3, replace=False)
                r0, r1, r2 = r

                # ---- mutation (DE/rand/1) ----
                mutant = pop[r0] + F_new * (pop[r1] - pop[r2])

                # ---- binomial crossover ----
                j_rand = np.random.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if np.random.rand() < CR_new or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i][j]

                # ---- boundary handling (reflection) ----
                # reflect out-of-bounds coordinates
                below = trial < lb
                if below.any():
                    trial[below] = 2.0 * lb[below] - trial[below]
                above = trial > ub
                if above.any():
                    trial[above] = 2.0 * ub[above] - trial[above]
                # final clipping (in case reflection overshoots)
                trial = np.clip(trial, lb, ub)

                # ---- evaluation ----
                trial_y = func(trial)
                evals_used += 1

                # ---- greedy selection ----
                if trial_y < y[i]:
                    pop[i] = trial
                    y[i] = trial_y
                    # keep the successful parameters
                    F[i] = F_new
                    CR[i] = CR_new
                    if trial_y < best_y:
                        best_x = trial.copy()
                        best_y = trial_y

        return best_x, best_y
