# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) algorithm for black-box minimization.
#          Uses the classic DE/rand/1/bin mutation strategy with a dithering scale factor.
#          Population size is set proportionally to the evaluation budget to balance exploration
#          and exploitation. Boundary violations are handled by simple clipping.
# Search state: A population of candidate solutions (vectors) stored in a matrix, along with their
#               associated objective values. The best individual and its value are tracked.
# Candidate generation: For each target vector in the population, three distinct random vectors
#                       are selected. A mutant vector is formed as base + F * (diff1 - diff2).
#                       F is sampled per-mutation from a uniform distribution in [0.5, 1.0) (dithering).
#                       Crossover (binomial) combines the mutant with the target according to crossover rate CR.
# Selection and replacement: Greedy selection: trial replaces target if it yields a lower objective value.
# Adaptation: The scale factor F and crossover rate CR are static hyperparameters (default: F_baseline=0.5,
#             F_range=0.5, CR=0.9). No online adaptation is used to keep code simple.
# Exploration mechanisms: Random mutation and crossover promote diversity; dithering provides varying step sizes;
#                         the population maintains multiple search points.
# Exploitation mechanisms: The greedy selection drives improvement toward better regions; the base vector is
#                          randomly chosen, so exploitation is moderate; the method is more explorative than
#                          DE/best/1.
# Boundary handling: Clipping: components of the trial vector outside [lb, ub] are set to the nearest bound.
# Budget strategy: The population size is computed as max(4, min(50, int(budget / 10))) to ensure enough
#                  generations (at least ~10 generations when budget≥40). For very small budgets (<40),
#                  the population size is reduced to 4 and the number of generations is limited accordingly.
# Closest known influences: Classic DE/rand/1/bin as described by Storn and Price (1997) with dithering
#                           (also known as jitter or random F).
# Novelty or unusual aspects: None; this is a straightforward, minimal implementation suitable for a benchmark
#                             harness. The population size adaptation to budget avoids premature termination.
# Failure modes: May stagnate on highly multimodal landscapes if population diversity disappears; may converge
#                slowly on separable or smooth functions because of random base selection; boundary clipping
#                can lead to repeated identical trial vectors if many components are near bounds.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

        # Set population size based on budget to guarantee at least ~10 generations.
        pop = int(budget / 10)
        if pop < 4:
            pop = 4
        elif pop > 50:
            pop = 50
        self.pop_size = pop

        # Hyperparameters for DE/rand/1/bin with dithering.
        self.F_base = 0.5        # base scale factor
        self.F_range = 0.5       # dithering range: F in [F_base, F_base+F_range)
        self.CR = 0.9            # crossover probability

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise ValueError("Objective function does not provide bounds.")

        dim = self.dim
        pop_size = self.pop_size
        max_evals = self.budget

        # Initialise population uniformly in bounds.
        pop = np.random.uniform(lb, ub, size=(pop_size, dim))
        # Evaluate all initial individuals.
        fitness = np.array([func(x) for x in pop])
        evals = pop_size

        # Track best.
        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop: run until budget exhausted.
        while evals < max_evals:
            # Determine number of trial vectors we can afford this generation.
            # We generate trial for each target, but limit by remaining budget.
            trials_possible = min(pop_size, max_evals - evals)
            if trials_possible <= 0:
                break

            # We'll iterate over a random subset or all targets but stop when evals exhausted.
            for i in range(trials_possible):
                # Choose three distinct random indices different from i.
                idxs = [idx for idx in range(pop_size) if idx != i]
                if len(idxs) < 3:
                    # population too small, skip this target
                    continue
                a, b, c = np.random.choice(idxs, size=3, replace=False)

                # Mutation with dithering: F drawn uniformly in [F_base, F_base+F_range)
                F = self.F_base + np.random.rand() * self.F_range
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary handling: clip to bounds
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_fit = func(trial)
                evals += 1

                # Selection
                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()

            # If for some reason we cannot generate any more trials (e.g., pop_size=0), break.
            if trials_possible < 1:
                break

        # Return best found solution
        return best_x, best_y
