import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE/rand/1/bin) with dither and restarts.
#          Designed for black-box minimization on the GNBG benchmark across all dimensions.
# Search state: A population of candidate solutions (array of shape (NP, dim)) and their
#               fitness values. The best solution and its fitness are tracked.
# Candidate generation: For each target vector, a donor is created as the base vector plus
#                       a scaled difference of two random distinct population members.
#                       The scaling factor F is sampled uniformly in [0.5, 1.0] per generation (dither).
#                       Binomial crossover combines donor and target with probability CR.
# Selection and replacement: Greedy – trial replaces target if its fitness is better.
# Adaptation: Scaling factor F is uniformly random in each generation to balance exploration.
#             Crossover rate CR is fixed at 0.9.
# Exploration mechanisms: Dither on F, random population initialization, binomial crossover
#                         that can mix components from any parent.
# Exploitation mechanisms: Differential mutation using current population members,
#                          greedy selection favoring improvement, and tracking the best.
# Boundary handling: Components are clipped to [lb, ub] after mutation and crossover.
# Budget strategy: The population size NP is scaled logarithmically with dimension
#                  (min 4, max 30). The algorithm runs generation by generation until
#                  the remaining budget is insufficient for a full generation.
# Closest known influences: Standard DE/rand/1/bin with dither (e.g., Storn & Price, 1997).
# Novelty or unusual aspects: Very compact implementation; no explicit restart mechanism;
#                             budget tracking uses a while loop that stops when < NP evaluations remain.
# Failure modes: May converge prematurely on highly multimodal landscapes if NP is too small.
#                Clip-based boundary handling can cause stagnation near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        """
        Initialize the optimizer.

        Parameters
        ----------
        budget : int
            Maximum number of objective function evaluations.
        dim : int
            Dimensionality of the problem.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Run the optimization.

        Parameters
        ----------
        func : callable
            Objective function to minimize. Must expose bounds via either
            `func.lower` / `func.upper` or `func.bounds.lb` / `func.bounds.ub`.

        Returns
        -------
        best_x : np.ndarray
            Best solution found.
        best_y : float
            Corresponding objective value.
        """
        # ---- Extract bounds -------------------------------------------------
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)
        else:
            # fallback – unlikely to be needed
            lb = np.full(self.dim, -1e10)
            ub = np.full(self.dim, 1e10)

        # ---- Parameters ----------------------------------------------------
        # Population size: logarithmic scaling, at least 4, at most 30.
        NP = max(4, min(30, int(4 + 3 * np.log(self.dim + 1))))
        CR = 0.9                     # crossover probability
        # remaining budget (will be decremented)
        remaining = self.budget

        # ---- Initialization ------------------------------------------------
        population = lb + (ub - lb) * np.random.uniform(0, 1, (NP, self.dim))
        fitness = np.full(NP, np.inf)

        for i in range(NP):
            if remaining <= 0:
                break
            fitness[i] = func(population[i])
            remaining -= 1

        # track best
        best_idx = np.argmin(fitness)
        best_x = population[best_idx].copy()
        best_y = fitness[best_idx]

        # ---- Main DE loop --------------------------------------------------
        while remaining >= NP:
            # dither: random F per generation
            F = 0.5 + 0.5 * np.random.uniform()

            # indices for each target vector (0..NP-1)
            indices = np.arange(NP)
            # for each target, pick three distinct random vectors different from itself
            for i in range(NP):
                # random indices for base, difference vectors
                candidates = np.delete(indices, i)
                np.random.shuffle(candidates)
                r1, r2, r3 = candidates[:3]

                # mutation
                donor = population[r1] + F * (population[r2] - population[r3])

                # binomial crossover
                cross_points = np.random.uniform(0, 1, self.dim) < CR
                # at least one component comes from donor
                if not np.any(cross_points):
                    cross_points[np.random.randint(self.dim)] = True
                trial = np.where(cross_points, donor, population[i])

                # boundary clipping
                trial = np.clip(trial, lb, ub)

                # evaluate
                if remaining <= 0:
                    break
                trial_fitness = func(trial)
                remaining -= 1

                # greedy selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_y:
                        best_y = trial_fitness
                        best_x = trial.copy()

        # ---- Final best result ---------------------------------------------
        return best_x, best_y
