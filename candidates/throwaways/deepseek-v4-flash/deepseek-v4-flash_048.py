# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) optimizer for black-box minimization.
#          Uses the classic DE/rand/1/bin strategy with fixed parameters.
# Search state: A population of candidate vectors (size NP) and the best solution found.
# Candidate generation: For each parent, a mutant vector is formed by adding the scaled
#                       difference of two other random population members to a third.
#                       Binomial crossover mixes the mutant with the parent.
# Selection and replacement: Each trial replaces the parent if it yields a lower (better)
#                            objective value (greedy selection).
# Adaptation: No adaptation; F = 0.8, CR = 0.9 are fixed across runs.
# Exploration mechanisms: Differential mutation and random index selection promote
#                         exploration of the search space.
# Exploitation mechanisms: As the population converges, mutation step sizes shrink
#                          naturally, focusing the search locally. Greedy selection
#                          retains improving solutions.
# Boundary handling: Infringing coordinates are clamped to the lower/upper bounds.
# Budget strategy: Every function evaluation is counted; the algorithm stops when the
#                  budget is exhausted, returning the best solution found so far.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None; a straightforward implementation suitable for
#                             benchmarking.
# Failure modes: Fixed parameters may not suit all landscapes; may stagnate on
#                highly multimodal or deceptive functions; population size tuned for
#                efficiency across dimensions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """Black-box minimizer using Differential Evolution (DE/rand/1/bin)."""

    def __init__(self, budget: int, dim: int):
        """
        Args:
            budget: Maximum number of function evaluations.
            dim: Dimensionality of the problem.
        """
        self.budget = budget
        self.dim = dim

        # DE parameters (fixed)
        # Population size: at least 5 and at most budget//2, scaled with dimension
        self.NP = max(5, min(budget // 2, 4 * dim))
        self.F = 0.8          # scaling factor
        self.CR = 0.9         # crossover probability

    def __call__(self, func):
        """Run the algorithm on a given objective function.

        Args:
            func: Callable object with either .lower / .upper or .bounds.lb / .bounds.ub
                  attributes providing the search bounds.

        Returns:
            (best_x, best_y): Best found solution and its objective value.
        """
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("func must provide bounds via .lower/.upper or .bounds.lb/.bounds.ub")

        dim = self.dim
        NP = self.NP
        F = self.F
        CR = self.CR
        budget = self.budget

        # Ensure population size does not exceed budget
        if NP > budget:
            NP = budget

        # Initialize population uniformly in the search space
        pop = np.random.uniform(lb, ub, (NP, dim))
        # Evaluate initial population
        evals = 0
        fitness = np.empty(NP)
        for i in range(NP):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1

        # Determine best so far
        best_idx = np.argmin(fitness[:evals])
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Main DE loop
        while evals < budget:
            new_pop = np.empty_like(pop)
            new_fit = np.empty(NP)
            # Generate offspring for each target vector
            for i in range(NP):
                if evals >= budget:
                    break

                # Choose three distinct random indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, size=3, replace=False)

                # Mutation: DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])

                # Binomial crossover
                cross_points = np.random.rand(dim) < CR
                # Ensure at least one dimension is crossed
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Boundary clipping
                trial = np.clip(trial, lb, ub)

                # Evaluate trial
                trial_y = func(trial)
                evals += 1

                # Greedy selection
                if trial_y < fitness[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_y
                    # Update best solution
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fit[i] = fitness[i]

            # Replace population (only if we produced all offspring)
            if evals < budget:
                pop = new_pop
                fitness = new_fit

        return best_x, best_y
