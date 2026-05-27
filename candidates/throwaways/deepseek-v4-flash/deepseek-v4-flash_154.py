import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: (mu, lambda)-Evolution Strategy with global intermediate recombination,
#   isotropic Gaussian mutation, and 1/5 success rule step-size adaptation.
# Search state: A population of mu candidate solutions (real vectors) and a scalar
#   global step-size sigma (relative to the per-dimension bound range).
# Candidate generation: Each generation, lambda offspring are created by adding
#   isotropic Gaussian noise (scaled by sigma and the bound range) to the
#   population mean (global recombination). Offspring are clipped to the bounds.
# Selection and replacement: Comma selection: the mu offspring with the lowest
#   objective values become the new parent population.
# Adaptation: The step-size sigma is updated every generation using the 1/5
#   success rule over a sliding window. If the fraction of offspring that
#   improve upon the current best exceeds 0.2, sigma is increased by a factor;
#   otherwise it is decreased.
# Exploration mechanisms: The initial sigma is set to 0.2 of the domain range,
#   and the success rule encourages larger steps when many improvements occur.
# Exploitation mechanisms: Recombination to the population mean and comma
#   selection focus the search on promising regions. Decreasing sigma refines
#   the search when improvements become rare.
# Boundary handling: Offspring coordinates are clipped to the function bounds
#   after mutation.
# Budget strategy: The algorithm computes the maximum number of generations
#   that can be performed given the population sizes and the evaluation budget.
#   At each generation, the number of offspring is limited to the remaining
#   evaluations. The loop stops as soon as the budget is exhausted.
# Closest known influences: Classical Evolution Strategy (Rechenberg, Schwefel).
# Novelty or unusual aspects: None.
# Failure modes: May struggle with highly multimodal or ill‑conditioned
#   landscapes due to the isotropic mutation and the simple global recombination.
#   Low budgets may prevent the algorithm from converging.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """(mu, lambda)-Evolution Strategy with global recombination and 1/5 success rule."""
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ---------- bounds ----------
        try:
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        except AttributeError:
            lb = np.array(func.bounds.lb, dtype=float)
            ub = np.array(func.bounds.ub, dtype=float)
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)
        ranges = ub - lb

        # ---------- ES parameters ----------
        mu = max(1, 4 + int(3 * np.log(self.dim)))          # parent population
        lam = 4 * mu                                        # offspring per generation
        sigma = 0.2                                         # initial relative step size

        # ---------- initialisation ----------
        pop = np.random.uniform(lb, ub, (mu, self.dim))
        fit = np.array([func(x) for x in pop])
        evals = mu

        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # step-size adaptation state
        window = 10
        success_hist = []           # list of success rates for recent generations

        # ---------- main loop ----------
        while evals < self.budget:
            # how many offspring can we afford in this generation?
            remaining = self.budget - evals
            lam_actual = min(lam, remaining)
            if lam_actual <= 0:
                break

            old_best = best_y               # snapshot before generation

            # global recombination (mean of parents)
            mean = np.mean(pop, axis=0)

            # generate offspring
            off = np.empty((lam_actual, self.dim))
            for i in range(lam_actual):
                step = sigma * ranges * np.random.randn(self.dim)
                off[i] = mean + step
                np.clip(off[i], lb, ub, out=off[i])

            # evaluate offspring
            off_fit = np.array([func(x) for x in off])
            evals += lam_actual

            # update global best
            for i in range(lam_actual):
                if off_fit[i] < best_y:
                    best_y = off_fit[i]
                    best_x = off[i].copy()

            # comma selection: keep mu best offspring
            if lam_actual >= mu:
                order = np.argsort(off_fit)[:mu]
                pop = off[order]
                fit = off_fit[order]
            else:
                # not enough offspring – keep all (mu shrinks temporarily)
                order = np.argsort(off_fit)[:lam_actual]
                pop = off[order]
                fit = off_fit[order]
                mu = lam_actuall   # adjust for next iteration (will be overwritten by budget check)

            # success rate for step-size adaptation
            n_success = np.sum(off_fit < old_best)
            success_hist.append(n_success / lam_actual)

            if len(success_hist) >= window:
                avg_success = np.mean(success_hist[-window:])
                if avg_success > 0.2:
                    sigma *= np.exp(0.1)
                else:
                    sigma *= np.exp(-0.1)
                success_hist = success_hist[-window:]  # keep only window

        return best_x, best_y
