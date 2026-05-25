import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Self-adaptive differential evolution (jDE) for black-box minimization.
# Search state: Population of candidate solutions (real vectors) and their fitness values.
# Candidate generation: For each parent, a mutant vector is created by adding a scaled difference between two random population members to a third. Then binomial crossover combines the parent and mutant using a per‑individual crossover probability.
# Selection and replacement: Greedy – a trial replaces the parent if it has lower (better) fitness.
# Adaptation: Each individual carries its own scaling factor F and crossover rate CR. With probability tau (0.1) they are re‑sampled from uniform distributions; otherwise they persist. This adapts the search to the local landscape.
# Exploration mechanisms: Random initialization over the whole domain; mutation with random donor selection; periodic reset of F/CR keeps diversity.
# Exploitation mechanisms: Crossover combines good solutions; greedy selection preserves improvements; decreasing component differences when population converges.
# Boundary handling: Feasible components of the trial vector are clipped to the bounds.
# Budget strategy: Exact evaluation count is tracked; algorithm terminates when reaching the budget.
# Closest known influences: jDE (Brest et al., 2006) – a standard self‑adaptive differential evolution variant.
# Novelty or unusual aspects: None – a straightforward implementation of a well‑known adaptive DE.
# Failure modes: May converge prematurely if population collapses to a narrow region; low budget (<< 20*dim) degrades performance; highly multimodal landscapes may need larger population.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    """Self‑adaptive Differential Evolution (jDE) for black‑box minimization."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # Population size: fixed at 20 for robustness across dimensions and budgets.
        self.np = 20
        # Adaptation parameters (jDE defaults)
        self.tau1 = 0.1     # probability to update F
        self.tau2 = 0.1     # probability to update CR
        self.F_l = 0.1      # lower bound for F
        self.F_u = 0.9      # upper bound for F
        self.CR_l = 0.0     # lower bound for CR
        self.CR_u = 1.0     # upper bound for CR

    def __call__(self, func):
        # ---------- read bounds ----------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        elif hasattr(func, 'bounds'):
            if isinstance(func.bounds, tuple) and len(func.bounds) == 2:
                lb, ub = func.bounds
            else:
                lb = func.bounds.lb
                ub = func.bounds.ub
        else:
            raise AttributeError("Cannot read bounds from func.")
        # ensure arrays
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        NP = self.np
        D = self.dim
        budget = self.budget

        # ---------- initial population ----------
        # uniform random inside [lb, ub]
        pop = lb + (ub - lb) * np.random.rand(NP, D)
        fitness = np.empty(NP)
        evals = 0
        for i in range(NP):
            fitness[i] = func(pop[i])
            evals += 1
        # best so far
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # ---------- initialize F and CR per individual ----------
        F = np.random.uniform(self.F_l, self.F_u, NP)
        CR = np.random.uniform(self.CR_l, self.CR_u, NP)

        # ---------- main loop ----------
        while evals < budget:
            # number of trials we can still generate without exceeding budget
            trials_this_gen = min(NP, budget - evals)

            # indices of individuals that will be updated (first trials_this_gen)
            update_idx = np.arange(trials_this_gen)

            # --- adaptation of F and CR for these individuals ---
            r = np.random.rand(trials_this_gen)
            F_new = np.where(r < self.tau1,
                             np.random.uniform(self.F_l, self.F_u, trials_this_gen),
                             F[update_idx])
            F[update_idx] = F_new

            r = np.random.rand(trials_this_gen)
            CR_new = np.where(r < self.tau2,
                              np.random.uniform(self.CR_l, self.CR_u, trials_this_gen),
                              CR[update_idx])
            CR[update_idx] = CR_new

            # --- generate trial vectors ---
            for j in range(trials_this_gen):
                i = update_idx[j]

                # choose three distinct indices different from i
                candidates = np.setdiff1d(np.arange(NP), i)
                if len(candidates) < 3:
                    # Should never happen for NP >= 4
                    break
                r_idx = np.random.choice(candidates, size=3, replace=False)
                a, b, c = r_idx

                # mutation: DE/rand/1
                v = pop[a] + F[i] * (pop[b] - pop[c])

                # binomial crossover
                trial = pop[i].copy()
                jrand = np.random.randint(D)
                mask = np.random.rand(D) < CR[i]
                mask[jrand] = True   # ensure at least one component changes
                trial[mask] = v[mask]

                # boundary handling: clip to [lb, ub]
                trial = np.clip(trial, lb, ub)

                # evaluate
                f_trial = func(trial)
                evals += 1

                # selection: greedy
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    # update global best
                    if f_trial < best_y:
                        best_x = trial.copy()
                        best_y = f_trial

                # stop if budget exhausted
                if evals >= budget:
                    break

        return best_x, best_y
