import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A compact Differential Evolution (DE) minimizer that uses rand/1/bin strategy with static parameters.
# Search state: A population of candidate solutions with associated fitness values, plus the best found solution.
# Candidate generation: For each target vector, a mutant is created by adding the scaled difference of two random population
#   vectors to a third base vector (DE/rand/1). A trial vector is produced via binomial crossover with the target.
# Selection and replacement: Greedy selection: the trial replaces the target if its fitness is lower (minimization).
# Adaptation: No parameter adaptation; scale factor F=0.8 and crossover rate CR=0.9 are fixed.
# Exploration mechanisms: Mutation using random population vectors provides global exploration; crossover mixes components.
# Exploitation mechanisms: The fitness-based selection preserves better solutions; the population gradually converges.
# Boundary handling: Trial vectors are clipped component-wise to the function's lower and upper bounds.
# Budget strategy: The algorithm runs for a fixed number of function evaluations equal to the budget, then returns the best.
# Closest known influences: Classic Differential Evolution (Storn & Price, 1997).
# Novelty or unusual aspects: None; a straightforward, well-known stochastic optimizer chosen for robustness.
# Failure modes: May converge prematurely on highly multimodal landscapes if the population is too small or budget too low.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    """Minimizer using Differential Evolution with budget constraint."""

    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Cannot find bounds on func (need lower/upper or bounds.lb/bounds.ub)")

        # Ensure arrays are 1-D of correct dimension
        if lb.ndim == 0:
            lb = np.full(self.dim, lb)
        if ub.ndim == 0:
            ub = np.full(self.dim, ub)

        # Population size: at least 4, but not too large relative to budget
        popsize = max(4, min(10 * self.dim, self.budget // 4))
        # Initialize population uniformly in [lb, ub]
        pop = lb + (ub - lb) * np.random.rand(popsize, self.dim)
        # Evaluate population
        fits = np.array([func(x) for x in pop])
        eval_count = popsize
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_y = fits[best_idx]

        # DE parameters
        F = 0.8
        CR = 0.9

        # Main loop
        while eval_count < self.budget:
            for i in range(popsize):
                # Choose three distinct random indices different from i
                indices = list(range(popsize))
                indices.remove(i)
                a, b, c = np.random.choice(indices, size=3, replace=False)
                # Mutation: create donor vector
                donor = pop[a] + F * (pop[b] - pop[c])
                # Binomial crossover
                cross_mask = np.random.rand(self.dim) < CR
                # Ensure at least one component is from donor
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(self.dim)] = True
                trial = np.where(cross_mask, donor, pop[i])
                # Boundary handling: clip to bounds
                trial = np.clip(trial, lb, ub)
                # Evaluate trial
                trial_fit = func(trial)
                eval_count += 1
                # Selection
                if trial_fit < fits[i]:
                    pop[i] = trial
                    fits[i] = trial_fit
                    # Update global best
                    if trial_fit < best_y:
                        best_y = trial_fit
                        best_x = trial.copy()
                # Check budget after each evaluation
                if eval_count >= self.budget:
                    break
            # End inner loop – break out if budget exhausted
            if eval_count >= self.budget:
                break

        return best_x, best_y
