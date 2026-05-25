import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) optimiser for black-box
#          minimisation. It adapts population size to the available budget and
#          dimension, performs classic DE/rand/1/bin mutation and binomial
#          crossover, then selects between parent and offspring using a greedy
#          one-to-one replacement. Bounds are enforced by reflecting infeasible
#          coordinates back into the domain.
# Search state: A population of candidate solutions x (size popsize × dim)
#               and their corresponding objective values y.
# Candidate generation: For each target vector, three distinct random
#                       individuals are selected (mutually exclusive and
#                       different from the target) to form a mutant via
#                       F * (b - c) added to a. Then binomial crossover with
#                       probability CR swaps mutant components into a trial.
# Selection and replacement: If the trial vector has lower (better) objective
#                            value than the target, it replaces the target in
#                            the next generation; otherwise the target remains.
# Adaptation: No online parameter adaptation – F and CR are fixed. Population
#             size is set heuristically at initialisation based on dimension
#             and budget.
# Exploration mechanisms: Large scaling factor (F=0.8) and moderate crossover
#                         (CR=0.9) encourage broad exploration early.
#                         The random selection of base and difference vectors
#                         maintains population diversity.
# Exploitation mechanisms: Greedy selection retains only improvements, and
#                          as the population converges, difference vectors
#                          become smaller, naturally shifting towards local
#                          refinement.
# Boundary handling: Reflection off the bounds: if any coordinate of a trial
#                    vector lies outside [lb, ub], it is reflected symmetrically
#                    until it lies inside. This keeps solutions feasible.
# Budget strategy: The population size is set to max(10, min(20, dim*2, budget//5))
#                  to ensure a reasonable number of generations (at least 5).
#                  The loop runs generation by generation, evaluating one trial
#                  per target per generation, stopping exactly when the
#                  cumulative evaluation count reaches the budget.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997)
#                           with DE/rand/1/bin strategy and reflection boundary
#                           handling.
# Novelty or unusual aspects: None – this is a straightforward implementation
#                            designed for clarity and robustness.
# Failure modes: May need many evaluations (O(popsize*gens)) to converge;
#                works poorly on highly multimodal or deceptive landscapes with
#                very low budget. Fixed F and CR may be suboptimal for some
#                problems.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Differential Evolution minimiser for the GNBG benchmark."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # ----- read bounds -----
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            b = func.bounds
            lb = np.array(b.lb, dtype=float)
            ub = np.array(b.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds on func")

        dim = self.dim
        budget = self.budget

        # ----- population size heuristic -----
        # ensure we can run at least 5 generations; limit to 20 to keep per‑gen cost low
        popsize = max(10, min(20, dim * 2, budget // 5))
        # parameters (classic DE/rand/1/bin)
        F = 0.8
        CR = 0.9

        # ----- initialisation -----
        pop = np.random.uniform(lb, ub, size=(popsize, dim))
        y = np.array([func(p) for p in pop])
        evals = popsize
        best_idx = np.argmin(y)
        best_x = pop[best_idx].copy()
        best_y = y[best_idx]

        # ----- main loop -----
        while evals < budget:
            # one generation: one trial per individual
            for i in range(popsize):
                if evals >= budget:
                    break

                # pick three distinct random indices different from i
                candidates = list(range(popsize))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)

                # mutation / crossover
                j_rand = np.random.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = pop[a, j] + F * (pop[b, j] - pop[c, j])
                    else:
                        trial[j] = pop[i, j]

                # boundary reflection
                trial = self._reflect(trial, lb, ub)

                # evaluation
                fy = func(trial)
                evals += 1

                # selection (greedy one‑to‑one)
                if fy < y[i]:
                    pop[i] = trial
                    y[i] = fy
                    if fy < best_y:
                        best_x = trial.copy()
                        best_y = fy

        return best_x, best_y

    @staticmethod
    def _reflect(x, lb, ub):
        """Reflect coordinates outside [lb, ub] back into the domain."""
        out_low = x < lb
        out_high = x > ub
        # reflect until inside (multiple reflections if needed)
        while np.any(out_low) or np.any(out_high):
            x[out_low] = 2 * lb[out_low] - x[out_low]
            x[out_high] = 2 * ub[out_high] - x[out_high]
            out_low = x < lb
            out_high = x > ub
        return x
