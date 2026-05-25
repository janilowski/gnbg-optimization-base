import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Differential Evolution (DE/rand/1/bin) for black-box minimization.
# Search state: A population of NP candidate vectors stored in self.pop (NP x dim)
#               and their corresponding fitness values in self.fitness (NP,).
#               The best individual and its fitness are tracked (self.best_x, self.best_y).
# Candidate generation: For each population member, a trial vector is created by
#                       mutation (base + F * (donor - donor2)) followed by binomial
#                       crossover with the current member, using probability CR.
# Selection and replacement: Greedy selection: trial replaces the current member if
#                            its fitness is lower (minimization).
# Adaptation: F (differential weight) and CR (crossover rate) are fixed parameters.
# Exploration mechanisms: Random mutation based on three distinct population members
#                         provides exploration. Crossover mixes trial with parent,
#                         maintaining diversity.
# Exploitation mechanisms: The best solution is carried over if not replaced; the
#                          population tends to converge as better solutions are kept.
# Boundary handling: Trial vectors are clipped to the variable bounds [lb, ub].
# Budget strategy: Population is evaluated once at start, then each trial uses one
#                  function evaluation if created. Number of generations is limited
#                  by remaining budget: we stop when remaining budget < population size
#                  to avoid incomplete evaluations. After the last full generation,
#                  remaining evaluations can be used for a few local improvements
#                  (not implemented for simplicity; could add random restarts).
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None; straightforward implementation.
# Failure modes: High-dimensional or highly multimodal landscapes may require
#                larger population size or adaptive parameters. Fixed parameters
#                may be suboptimal for some problems.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # Population size: a heuristic, at least 4, at most budget/2
        self.NP = max(4, min(budget // 2, 10 * dim))
        # Differential weight and crossover rate (typical values)
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        # Retrieve bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds from func")

        # Ensure bounds are 1D arrays
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
            ub = np.full(self.dim, ub)
        elif lb.ndim == 1 and lb.shape[0] == self.dim:
            pass
        else:
            raise ValueError("Bounds shape mismatch")

        # Initialize population uniformly in [lb, ub]
        pop = np.random.uniform(lb, ub, (self.NP, self.dim))
        # Evaluate entire population
        fitness = np.array([func(x) for x in pop])
        eval_count = self.NP

        # Track best
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        while True:
            # Estimate remaining generations: each generation uses NP evaluations
            if eval_count + self.NP > self.budget:
                # Not enough budget for a full generation; stop
                break

            # Generate next generation
            for i in range(self.NP):
                # Mutation: pick three distinct indices different from i
                idxs = [j for j in range(self.NP) if j != i]
                chosen = np.random.choice(idxs, 3, replace=False)
                a, b, c = pop[chosen[0]], pop[chosen[1]], pop[chosen[2]]
                mutant = a + self.F * (b - c)

                # Crossover (binomial)
                trial = pop[i].copy()
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary handling: clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_fitness = func(trial)
                eval_count += 1

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

            # After generation, check if budget exhausted exactly
            if eval_count >= self.budget:
                break

        return best_x, best_y
