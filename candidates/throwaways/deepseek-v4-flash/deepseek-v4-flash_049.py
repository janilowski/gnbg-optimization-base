import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE/rand/1/bin) with dithering and a final random‑search phase.
# Search state: A population of candidate vectors (size NP) and the best solution found so far.
# Candidate generation: For each target vector, a mutant is created as base + F * (diff1 - diff2)
#   where F is uniformly sampled in [0.5, 1.0] per mutation (dither). The trial vector is formed by
#   binomial crossover with probability CR = 0.9.
# Selection and replacement: Greedy – the trial replaces the target if and only if its objective value is lower.
# Adaptation: No explicit adaptation; dither provides a simple diversity mechanism.
# Exploration mechanisms: Population diversity, random target selection for mutation, dithering, crossover.
# Exploitation mechanisms: Greedy selection drives the population toward low‑fitness regions;
#   the final random search is pure exploration but may locate a slightly better point.
# Boundary handling: All coordinates are clamped to the search bounds.
# Budget strategy: The budget is divided into full DE generations (each using NP evaluations) followed by
#   isolated uniform random trials. If the remaining budget is not enough for a full generation, random
#   search consumes the remainder.
# Closest known influences: Storn & Price (1997) differential evolution; dithering is a common variant.
# Novelty or unusual aspects: The simple post‑cycle uniform random search is a straightforward way
#   to exhaust the budget while retaining a chance to find better solutions.
# Failure modes: Premature convergence on highly multimodal landscapes when NP is too small; poor
#   performance on extremely ill‑conditioned problems due to isotropic mutation.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

        # Population size: at least 5, at most floor(budget/3), but not less than 3
        # and scaled by dimension (4*dim) to maintain diversity.
        self.NP = max(5, min(budget // 3, 4 * dim))
        self.NP = max(self.NP, 3)                # need at least 3 vectors for mutation
        if self.NP > budget:
            self.NP = budget                     # cap so at least one generation can run

        self.CR = 0.9
        self.rng = np.random.default_rng()
        self.lower = None
        self.upper = None

    def _read_bounds(self, func):
        """Extract lower and upper bounds from the objective function object."""
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            self.lower = np.array(func.lower, dtype=float)
            self.upper = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            if hasattr(b, 'lb') and hasattr(b, 'ub'):
                self.lower = np.array(b.lb, dtype=float)
                self.upper = np.array(b.ub, dtype=float)
            elif hasattr(b, 'lower') and hasattr(b, 'upper'):
                self.lower = np.array(b.lower, dtype=float)
                self.upper = np.array(b.upper, dtype=float)
            else:
                # fallback: assume bounds from tuple (lb, ub)
                self.lower = np.array(b[0], dtype=float)
                self.upper = np.array(b[1], dtype=float)
        else:
            raise AttributeError("Cannot read bounds from func; need .lower/.upper or .bounds.lb/.ub")

    def __call__(self, func):
        self._read_bounds(func)
        dim = self.dim
        lower, upper = self.lower, self.upper
        NP = self.NP
        budget = self.budget

        # ---- initialisation ----
        pop = self.rng.uniform(lower, upper, size=(NP, dim))
        # evaluate population
        fits = np.array([func(x) for x in pop])
        budget -= NP

        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_y = fits[best_idx]

        # ---- main DE loop ----
        while budget >= NP:
            for i in range(NP):
                # choose three distinct indices different from i
                candidates = [j for j in range(NP) if j != i]
                a, b, c = self.rng.choice(candidates, size=3, replace=False)
                base, diff1, diff2 = pop[a], pop[b], pop[c]

                # mutation with dither
                F = 0.5 + 0.5 * self.rng.random()
                mutant = base + F * (diff1 - diff2)

                # binomial crossover
                trial = pop[i].copy()
                j_rand = self.rng.integers(dim)
                for j in range(dim):
                    if self.rng.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # boundary clamping
                trial = np.clip(trial, lower, upper)

                # evaluation
                fit_trial = func(trial)
                budget -= 1

                # selection
                if fit_trial < fits[i]:
                    pop[i] = trial
                    fits[i] = fit_trial
                    if fit_trial < best_y:
                        best_y = fit_trial
                        best_x = trial.copy()

        # ---- exhaust remaining budget with uniform random sampling ----
        while budget > 0:
            x = self.rng.uniform(lower, upper)
            y = func(x)
            budget -= 1
            if y < best_y:
                best_y = y
                best_x = x

        return best_x, best_y
