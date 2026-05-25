import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Differential Evolution (DE) variant with random dithering of scale factor F and crossover rate CR,
#          reflective boundary handling, and a fallback local search using residual budget.
# Search state: A population of candidate vectors and their associated fitness values. The best known solution is tracked separately.
# Candidate generation: For each population member, a mutant is formed using the "DE/rand/1/bin" scheme: base = random individual,
#          difference = two other random distinct individuals, scaled by F. Binomial crossover with probability CR combines
#          mutant and target to produce a trial vector. F is sampled per trial from a truncated Cauchy(0.5,0.1) distribution,
#          CR from Uniform(0,1).
# Selection and replacement: Greedy, synchronous replacement: all trial vectors are evaluated before updating the population.
#          A trial replaces its target only if it has better (lower) fitness.
# Adaptation: F and CR are resampled for each trial independently, providing parameter diversity and implicit adaptation.
# Exploration mechanisms: Random selection of base/difference vectors, high average crossover probability, and dithering F
#          encourage wide exploration. Boundary reflection prevents loss of diversity at the domain edges.
# Exploitation mechanisms: Greedy selection focuses the population around promising regions. The optional final local search
#          performs Gaussian perturbations of the best solution using any unused evaluations.
# Boundary handling: Out-of-bound coordinates are reflected inward (mirror effect), then clipped to guarantee feasibility.
# Budget strategy: Population size NP = min(50, max(4, budget//2)). Generations = floor((budget - NP) / NP). Any remaining
#          evaluations (if budget – NP – generations*NP > 0) are used for a local random perturbation of the best solution.
# Closest known influences: Classic Differential Evolution (Storn & Price 1997) and the jDE self-adaptive variant.
# Novelty or unusual aspects: Very simple parameterisation; no complex population-size adaptation or restart schemes.
# Failure modes: May stagnate on multimodal landscapes with small population; no explicit restart. High-dimensional problems
#          may require larger NP than allowed by a tight budget.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Black-box minimizer using Differential Evolution with dithering and a final local search."""
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # --- Extract bounds -------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=np.float64)
            ub = np.asarray(func.upper, dtype=np.float64)
        else:
            lb = np.asarray(func.bounds.lb, dtype=np.float64)
            ub = np.asarray(func.bounds.ub, dtype=np.float64)

        budget = self.budget
        dim = self.dim
        if budget <= 0:
            return None, None

        best_x = None
        best_y = float('inf')
        evals = 0

        # --- Population size ------------------------------------------------
        NP = max(4, min(50, budget // 2))
        if budget < NP:
            # Not enough budget for a full population → pure random search
            NP = budget
            for _ in range(NP):
                x = np.random.uniform(lb, ub)
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # --- Initialisation -------------------------------------------------
        pop = np.random.uniform(lb, ub, size=(NP, dim))
        fitness = np.empty(NP)
        for i in range(NP):
            fitness[i] = func(pop[i])
            evals += 1
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # --- Main DE loop ---------------------------------------------------
        while evals + NP <= budget:
            # Generate trial vectors
            trials = np.empty_like(pop)
            trial_fitness = np.empty(NP)
            for i in range(NP):
                # Select three distinct indices different from i
                candidates = [j for j in range(NP) if j != i]
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                # Random dithering parameters
                F = np.random.standard_cauchy() * 0.1 + 0.5
                F = np.clip(F, 0.0, 1.0)
                CR = np.random.uniform(0.0, 1.0)
                # Mutant
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Binomial crossover
                trial = pop[i].copy()
                j0 = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j0:
                        trial[j] = mutant[j]
                # Reflection boundary handling
                trial = np.where(trial < lb, 2.0 * lb - trial, trial)
                trial = np.where(trial > ub, 2.0 * ub - trial, trial)
                trial = np.clip(trial, lb, ub)
                trials[i] = trial

            # Evaluate all trials
            for i in range(NP):
                trial_fitness[i] = func(trials[i])
                evals += 1
                if evals >= budget:   # safety (should not happen due to while condition)
                    break
            # Synchronous selection
            for i in range(NP):
                if trial_fitness[i] < fitness[i]:
                    pop[i] = trials[i]
                    fitness[i] = trial_fitness[i]
                    if trial_fitness[i] < best_y:
                        best_y = trial_fitness[i]
                        best_x = trials[i].copy()

        # --- Use leftover evaluations for local search around the best -------
        if evals < budget:
            sigma = (ub - lb) * 0.1   # 10% of the range
            while evals < budget:
                x = best_x + np.random.normal(0, sigma, size=dim)
                x = np.clip(x, lb, ub)
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x.copy()

        return best_x, best_y
