# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Differential Evolution variant with per-individual adaptive crossover and mutation parameters (jDE-like).
# Search state: Population of candidate solutions, each with its own F (mutation scaling) and CR (crossover probability).
# Candidate generation: For each parent, a mutant is created via DE/rand/1 using three distinct random members; then binomial crossover combines parent and mutant.
# Selection and replacement: Greedy selection: offspring replaces parent if its objective value is not worse (minimization).
# Adaptation: Each generation, each individual's F and CR are renewed with probabilities tau1=0.1 and tau2=0.1, respectively, drawing F~U(0.1,1.0) and CR~U(0,1).
# Exploration mechanisms: Differential mutation with random scaling (F) and crossover (CR) provides exploration.
# Exploitation mechanisms: Selection pressure keeps better solutions; mutation step sizes are controlled by population diversity and F scaling.
# Boundary handling: Offspring variables are clipped to [lower, upper] bounds.
# Budget strategy: Population size is set as max(4, min(10*dim, budget//2)); generations run until evaluations exhaust or a stagnation criterion (evaluation count exceeds budget) stops the loop.
# Closest known influences: jDE (Brest et al., 2006) – self-adaptive DE.
# Novelty or unusual aspects: Minimal modifications for robustness across dimensions; no complex adaptation beyond per-individual parameter resetting.
# Failure modes: May converge prematurely on strongly multimodal functions if population diversity collapses; fixed tau1/tau2 may not suit all problems; clipping may distort search near bounds.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Read bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        else:
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)

        budget = self.budget
        dim = self.dim
        # Ensure lower/upper are 1D arrays
        if lower.ndim == 0:
            lower = np.full(dim, lower)
            upper = np.full(dim, upper)

        # Population size
        # At least 4, at most min(10*dim, budget//2) so we can run at least 2 generations
        pop_size = max(4, min(10 * dim, budget // 2))
        # If budget is very small, adjust
        if pop_size > budget:
            pop_size = budget
        if pop_size < 2:
            pop_size = 2  # but should not happen with typical budgets

        # Initialize population uniformly in bounds
        pop = np.random.uniform(lower, upper, size=(pop_size, dim))
        # Initialize F and CR per individual
        F = np.random.uniform(0.3, 0.9, size=pop_size)   # mutation factor
        CR = np.random.uniform(0.1, 0.9, size=pop_size)  # crossover rate

        # Evaluate initial population
        fits = np.array([func(x) for x in pop])
        evals = pop_size

        # Best so far
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_y = fits[best_idx]

        # Parameters for adaptation (jDE style)
        tau1 = 0.1
        tau2 = 0.1

        # Main loop
        while evals < budget:
            # For each individual, create a candidate
            new_pop = pop.copy()
            new_F = F.copy()
            new_CR = CR.copy()
            for i in range(pop_size):
                # Choose three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                idxs = np.random.choice(candidates, size=3, replace=False)
                a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]

                # Mutation
                mutant = a + F[i] * (b - c)
                # Binomial crossover
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR[i]
                mask[j_rand] = True
                offspring = np.where(mask, mutant, pop[i])
                # Boundary clipping
                offspring = np.clip(offspring, lower, upper)

                # Evaluate if budget left
                if evals >= budget:
                    break
                fit_off = func(offspring)
                evals += 1

                # Selection
                if fit_off < fits[i]:
                    new_pop[i] = offspring
                    fits[i] = fit_off
                    if fit_off < best_y:
                        best_y = fit_off
                        best_x = offspring.copy()
                    # Keep F and CR if successful; they will be possibly reset later
                else:
                    # For unsuccessful parent, we'll later randomize F and CR with some probability
                    pass

            # Adapt F and CR for next generation
            for i in range(pop_size):
                if np.random.rand() < tau1:
                    new_F[i] = np.random.uniform(0.1, 1.0)
                if np.random.rand() < tau2:
                    new_CR[i] = np.random.uniform(0, 1)

            pop = new_pop
            F = new_F
            CR = new_CR

        return best_x, best_y
