import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE/rand/1/bin) minimizer designed
#          for black‑box problems.  It works well across moderate dimensions and
#          budgets, using only the Python standard library and NumPy.
# Search state: A population of candidate vectors stored in a NumPy array
#               (pop) and their corresponding fitness values (fitness).  The
#               best solution found so far is tracked separately.
# Candidate generation: For each parent, three distinct agents (a,b,c) are
#                       chosen uniformly from the population (excluding the
#                       parent).  A mutant is created as a + F * (b - c), then
#                       subject to binomial crossover with the parent to form
#                       the trial vector.
# Selection and replacement: Greedy – the trial replaces the parent if and
#                            only if its fitness (minimization) is strictly
#                            better.  The global best is updated accordingly.
# Adaptation: No online parameter adaptation; F and CR are fixed (0.7 and 0.9
#             respectively).  The population size is set once at construction
#             based on budget and dimension.
# Exploration mechanisms: Large initial population spread over the full domain,
#                         and the random differential mutation (difference
#                         vector) provides broad exploration.
# Exploitation mechanisms: As generations progress, the population contracts
#                          around good regions by the recombination and
#                          selection pressure.
# Boundary handling: Trial vectors are clipped componentwise to [lower, upper].
# Budget strategy: The initial population is evaluated once, then generations
#                  run until the evaluation budget is exhausted.  No extra
#                  evaluations beyond the budget are performed.
# Closest known influences: Standard DE/rand/1/bin (Storn & Price, 1997) with
#                           fixed parameters.
# Novelty or unusual aspects: None; the implementation is deliberately
#                             textbook‑simple to ensure robustness and clarity.
# Failure modes: (1) Very small budgets (< 4) do not allow the required number
#                of parents; the algorithm falls back to pure random search.
#                (2) Very high‑dimensional problems may require a much larger
#                population than the fixed heuristic can provide, but the
#                budget cap prevents excessive waste.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """
    Black‑box minimizer using Differential Evolution (DE/rand/1/bin).
    
    Parameters
    ----------
    budget : int
        Maximum number of function evaluations.
    dim : int
        Dimensionality of the problem.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # Population size heuristic: aim for about 10 generations, but never
        # exceed budget, dimension*10, or 200.  Must be at least 4 for DE.
        self.popsize = max(4, min(budget, int(budget / 10), 10 * dim, 200))
        # In case budget < 4, fall back to the budget itself (random search)
        self.popsize = min(self.popsize, budget)

    def __call__(self, func):
        # Read bounds (support both common API styles)
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            low = np.asarray(func.lower, dtype=float)
            high = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            low = np.asarray(func.bounds.lb, dtype=float)
            high = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find problem bounds")

        dim = self.dim
        budget = self.budget
        npop = self.popsize

        # If budget too small for DE, just do random search
        if npop < 4 or budget < 4:
            best_x = low + (high - low) * np.random.rand(dim)
            best_y = func(best_x)
            evals = 1
            while evals < budget:
                x = low + (high - low) * np.random.rand(dim)
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # === Initialise population and evaluate ===
        pop = low + (high - low) * np.random.rand(npop, dim)
        fitness = np.empty(npop)
        best_x = None
        best_y = np.inf
        for i in range(npop):
            fitness[i] = func(pop[i])
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = pop[i].copy()
        evals = npop

        # DE parameters (fixed)
        F = 0.7
        CR = 0.9

        # === Main evolution loop ===
        while evals < budget:
            for i in range(npop):
                if evals >= budget:
                    break

                # Select three distinct indices, all != i
                idxs = [j for j in range(npop) if j != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)

                # Mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Boundary repair: clip to bounds
                mutant = np.clip(mutant, low, high)

                # Binomial crossover
                jrand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[jrand] = mutant[jrand]  # ensure at least one diff

                # Evaluate trial
                trial_fit = func(trial)
                evals += 1

                # Selection (minimisation)
                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

        return best_x, best_y
