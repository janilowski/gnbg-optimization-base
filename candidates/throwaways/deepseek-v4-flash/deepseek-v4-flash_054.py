import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Simple Differential Evolution (DE/rand/1/bin) for continuous black-box minimization.
# Search state: A population of candidate solutions (list of arrays) and their objective values.
# Candidate generation: For each target vector, generate a donor via (a + F * (b - c)) using three distinct random population members; apply binomial crossover with probability CR to produce trial vector, ensuring at least one component comes from the donor.
# Selection and replacement: Greedy selection – replace the target if the trial has a lower (better) objective value.
# Adaptation: None; fixed parameters F = 0.5, CR = 0.9.
# Exploration mechanisms: Mutation with randomly scaled differences and crossover maintain diversity throughout the run.
# Exploitation mechanisms: The population contracts as better solutions dominate; greedy selection pushes towards local minima.
# Boundary handling: Trial vectors are clipped componentwise to the problem bounds.
# Budget strategy: The algorithm stops as soon as the number of function evaluations reaches the budget. The population size is set to max(4, min(10*dim, budget//5)) whenever possible, but is capped by the budget itself to ensure at least initial evaluations.
# Closest known influences: Standard Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None, straightforward implementation.
# Failure modes: May converge prematurely on highly multimodal landscapes; fixed parameters are not adapted to the problem; very small budgets make evolution ineffective and degenerate to random search.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Differential Evolution (DE/rand/1/bin) for black-box minimization."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # DE parameters
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            low = np.array(func.lower, dtype=float)
            high = np.array(func.upper, dtype=float)
        elif hasattr(func, 'bounds'):
            low = np.array(func.bounds.lb, dtype=float)
            high = np.array(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds from function object")

        dim = self.dim
        budget = self.budget

        # Determine population size
        if budget < 4:
            pop_size = budget  # degenerate case: each evaluation is one random point
        else:
            # scale with dimension, but not too small or large relative to budget
            pop_size = min(budget, max(4, min(10 * dim, budget // 5)))

        # Initialize population uniformly in bounds
        pop = np.random.uniform(low, high, size=(pop_size, dim))
        fit = np.empty(pop_size)
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            if evals >= budget:
                break
            fit[i] = func(pop[i])
            evals += 1
        best_idx = np.argmin(fit[:evals])
        best_x = pop[best_idx].copy()
        best_y = fit[best_idx]

        # Main DE loop (only if we have at least 3 individuals, else skip)
        if pop_size >= 4:
            while evals < budget:
                # Shuffle individual indices each generation to avoid bias
                order = np.random.permutation(pop_size)
                for i in order:
                    if evals >= budget:
                        break
                    # Mutation: choose three distinct indices different from i
                    candidates = [j for j in range(pop_size) if j != i]
                    a, b, c = np.random.choice(candidates, size=3, replace=False)
                    donor = pop[a] + self.F * (pop[b] - pop[c])

                    # Binomial crossover
                    trial = pop[i].copy()
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.rand() < self.CR or j == j_rand:
                            trial[j] = donor[j]

                    # Clip to bounds
                    trial = np.clip(trial, low, high)

                    # Evaluate trial
                    trial_fit = func(trial)
                    evals += 1

                    # Greedy selection
                    if trial_fit < fit[i]:
                        pop[i] = trial
                        fit[i] = trial_fit
                        if trial_fit < best_y:
                            best_y = trial_fit
                            best_x = trial.copy()

        # If budget was too small, we may still have some random points left;
        # otherwise we simply return the best found.
        return best_x, best_y
