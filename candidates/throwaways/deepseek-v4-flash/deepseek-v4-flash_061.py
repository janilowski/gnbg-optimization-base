import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) minimizer using the DE/rand/1/bin variant.
# Search state: A population of candidate solutions with their fitness values; best tracked separately.
# Candidate generation: For each target individual, a mutant is created by adding the scaled difference
#   of two random distinct population members to a third (base) member.
# Selection and replacement: Greedy – the offspring replaces the parent if it yields a lower (better) objective.
# Adaptation: The scaling factor F and crossover rate CR are fixed at 0.7 and 0.9 respectively; no dynamic adaptation.
# Exploration mechanisms: Differential mutation with random base and difference vectors promotes diversity.
# Exploitation mechanisms: Recombination (binomial crossover) blends mutant and parent; greedy selection drives convergence.
# Boundary handling: Offspring coordinates outside bounds are reflected back into the feasible region.
# Budget strategy: The algorithm runs generation by generation; total evaluations never exceed the given budget.
# Closest known influences: Classic DE (Storn & Price, 1997) with reflection boundary handling.
# Novelty or unusual aspects: None – standard textbook DE.
# Failure modes: May struggle on highly multimodal or non-separable functions due to fixed F, CR and lack of adaptation.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # Population size heuristic: at least 10, roughly 4*dim
        self.pop_size = max(10, 4 * dim)
        # Fixed control parameters
        self.F = 0.7       # differential weight
        self.CR = 0.9      # crossover probability
        self.rng = np.random.RandomState()  # seeded externally

    def __call__(self, func):
        # Determine bounds from func
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot locate bounds from func object")
        # Ensure lb/ub are 1D arrays
        lb = lb.flatten() if lb.ndim > 1 else lb
        ub = ub.flatten() if ub.ndim > 1 else ub

        pop_size = min(self.pop_size, self.budget)  # at least 1 solution
        if pop_size < 2:
            pop_size = 2

        # Initialize population uniformly within bounds
        pop = lb + (ub - lb) * self.rng.rand(pop_size, self.dim)
        # Evaluate initial population
        fit = np.array([func(x) for x in pop])
        evals = pop_size

        # Track best
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # Main DE loop
        while evals < self.budget:
            new_pop = np.empty_like(pop)
            new_fit = np.empty(pop_size)
            # For each target individual
            for i in range(pop_size):
                # Choose three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = self.rng.choice(candidates, size=3, replace=False)

                # Mutation (DE/rand/1)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])

                # Binomial crossover
                # Randomly pick a crossover point to ensure at least one coordinate changes
                j_rand = self.rng.randint(self.dim)
                trial = pop[i].copy()
                for j in range(self.dim):
                    if self.rng.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # Reflection boundary handling
                for j in range(self.dim):
                    if trial[j] < lb[j]:
                        trial[j] = 2 * lb[j] - trial[j]
                    elif trial[j] > ub[j]:
                        trial[j] = 2 * ub[j] - trial[j]
                # Clamp in case reflection still out of bounds (rare for symmetric bounds)
                trial = np.clip(trial, lb, ub)

                # Evaluate trial if budget allows
                if evals >= self.budget:
                    # Budget exhausted; break out of inner loop
                    # Store dummy values (not used)
                    new_pop[i] = trial
                    new_fit[i] = np.inf
                    continue

                trial_fit = func(trial)
                evals += 1

                # Greedy selection: replace if better or equal (to avoid stagnation)
                if trial_fit <= fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
                    # Update global best
                    if trial_fit < best_y:
                        best_x = trial.copy()
                        best_y = trial_fit
                else:
                    new_pop[i] = pop[i]
                    new_fit[i] = fit[i]

            # Replace population if we have full generation
            if evals >= self.budget:
                # Budget exceeded during generation, some entries are dummy.
                # We simply stop. The best is already tracked.
                break
            pop = new_pop
            fit = new_fit

        return best_x, best_y
