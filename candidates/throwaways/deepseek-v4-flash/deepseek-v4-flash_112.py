import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A simple classic Differential Evolution (DE/rand/1/bin) algorithm for continuous black-box minimization.
# Search state: A population of candidate solutions (vectors) stored in a numpy array, along with their objective values.
# Candidate generation: For each individual, a mutant vector is created as base + F * (diff1 - diff2), where base, diff1, diff2 are three distinct randomly selected population members. Binomial crossover mixes the mutant and the target with probability CR.
# Selection and replacement: Greedy selection: if the trial vector yields a better (or equal) objective value than the target, it replaces the target in the next generation.
# Adaptation: The scaling factor F and crossover rate CR are fixed constants (F = 0.8, CR = 0.9). No adaptive scheme is used.
# Exploration mechanisms: Mutation using random differential vectors provides diversity; binomial crossover mixes components.
# Exploitation mechanisms: Greedy replacement focuses search near better solutions; the population gradually converges.
# Boundary handling: Trial vectors that escape the feasible box are reflected back into the domain symmetrically.
# Budget strategy: The algorithm runs generation by generation until the cumulative number of function evaluations reaches the given budget. The population size is set dynamically based on budget and dimension to allow at least 10 generations if possible.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1995).
# Novelty or unusual aspects: No novelty – a clean, compact implementation of the standard DE/rand/1/bin variant.
# Failure modes: May converge prematurely on highly multimodal landscapes; static parameters may not suit all problems; performance degrades on very high dimensions; does not handle noise explicitly.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Obtain bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds. Expect func.lower/upper or func.bounds.lb/ub.")

        dim = self.dim
        # Enforce dimension consistency
        if lb.ndim == 0:
            lb = np.full(dim, lb)
            ub = np.full(dim, ub)
        else:
            lb = lb.flatten()[:dim]
            ub = ub.flatten()[:dim]

        # Ensure lb < ub
        lb = np.minimum(lb, ub)
        ub = np.maximum(lb, ub)

        # Dynamically set population size
        # Aim for at least 10 generations, but keep a minimum of 4 and a maximum of 50 (or budget/2)
        max_possible_pop = self.budget // 10
        pop_size = max(4, min(50, max_possible_pop))
        # But also make sure population size does not exceed budget/2 (otherwise too few generations)
        if pop_size > self.budget // 2:
            pop_size = max(4, self.budget // 2)
        # Never have more individuals than budget (for the initial eval)
        if pop_size > self.budget:
            pop_size = self.budget

        # DE parameters (fixed)
        F = 0.8        # scaling factor
        CR = 0.9       # crossover rate

        # Initialise population uniformly in the box
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        # Evaluate initial population
        fitness = np.array([func(x) for x in pop])   # shape (pop_size,)
        evals = pop_size
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        while evals < self.budget:
            # Determine how many trial vectors we can generate with remaining budget
            # (One evaluation per trial vector)
            remaining = self.budget - evals
            # Number of trial vectors this generation (cannot exceed pop_size)
            n_trials = min(pop_size, remaining)
            if n_trials <= 0:
                break

            # We'll process a random subset or the whole population? Process all if possible.
            # To keep it simple, we produce one trial per individual but stop early if budget exhausted.
            # We'll iterate over 'n_trials' random indices (or order) to avoid bias.
            order = np.random.permutation(pop_size)[:n_trials]

            for i in order:
                # Select three distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 3:
                    break   # should not happen with pop_size >=4
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # Mutation: base (a) + F * (b - c)
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                # Ensure at least one component from mutant
                trial[j_rand] = mutant[j_rand]

                # Boundary reflection
                # Reflect components outside [lb, ub] back inside
                low_viol = trial < lb
                high_viol = trial > ub
                trial[low_viol] = 2 * lb[low_viol] - trial[low_viol]
                trial[high_viol] = 2 * ub[high_viol] - trial[high_viol]
                # Clamp in case multiple reflections overshoot (rare but safe)
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_f = func(trial)
                evals += 1

                # Selection (greedy)
                if trial_f <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_f
                    if trial_f < best_y:
                        best_y = trial_f
                        best_x = trial.copy()

            # After processing trials (some may not have been processed if budget exactly filled)
            # No further adaptation; continue loop.

        return best_x, best_y
