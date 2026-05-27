# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a self-adaptive Differential Evolution (jDE) optimizer for
# black-box minimization. It uses a random initial population, generates trial vectors using
# DE/rand/1/bin with individual adaptation of scaling factor F and crossover rate CR, and
# applies a simple random repair for bound violations. The algorithm stops when the evaluation
# budget is exhausted, returning the best found solution and its objective value.
# Search state: A population of candidate solutions (x), objective values (y), and per-individual
# parameters (F, CR). The best solution and its value are tracked.
# Candidate generation: For each target vector from the current population, a mutant vector is
# created by adding the scaled difference of two random distinct population members to a third
# random distinct member (DE/rand/1). Then binomial crossover combines the target and mutant,
# using the individual's CR to decide which coordinates to take from the mutant.
# Selection and replacement: If the trial vector has lower objective value (minimization) than
# the target, it replaces the target in the next generation, and the target's parameters (F, CR)
# are replaced by the trial's parameters (which are generated anew with probability tau1/tau2).
# Otherwise the target and its parameters are retained.
# Adaptation: Each individual carries its own F and CR. At each generation, for each target,
# new parameters are generated from uniform distributions with probabilities tau1 and tau2
# (here 0.1). These parameters are used to generate the trial. If the trial succeeds, the
# new parameters are kept; otherwise the old ones are kept. This allows successful parameter
# values to spread.
# Exploration mechanisms: Differential mutation with random base vectors and difference vectors
# provides global exploration. The self-adaptive parameters allow the algorithm to adjust to
# the landscape.
# Exploitation mechanisms: The greedy selection (replace only if better) ensures convergence.
# The best solution is always tracked and returned.
# Boundary handling: If any coordinate of a trial vector falls outside the bounds, it is
# replaced with a uniformly random value within that coordinate's bounds.
# Budget strategy: The algorithm stops immediately when the allocated number of function
# evaluations (budget) is reached. The initial population consumes budget/2 evaluations, then
# the generation loop consumes the rest. The exact budget is never exceeded.
# Closest known influences: Classic Differential Evolution (DE/rand/1/bin) with self-adaptive
# parameters as in Brest et al. (2006) "Self-Adapting Control Parameters in Differential
# Evolution".
# Novelty or unusual aspects: None; standard implementation with careful budget management.
# Failure modes: Might perform poorly on highly multimodal or deceptive landscapes due to
# greedy selection and limited exploration when budget is very low. The self-adaptation may
# not converge quickly enough for very small budgets.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # jDE parameters
        self.tau1 = 0.1   # probability to regenerate F
        self.tau2 = 0.1   # probability to regenerate CR
        self.F_l = 0.1    # lower bound for F
        self.F_u = 0.9    # upper bound for F
        self.CR_l = 0.0   # lower bound for CR
        self.CR_u = 1.0   # upper bound for CR
        # population size: heuristic based on dimension, bounded by budget
        self.NP = max(4, min(100, int(budget * 0.1), 10 * dim))

    def __call__(self, func):
        # Extract bounds
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, 'bounds') and hasattr(func.bounds, 'lb') and hasattr(func.bounds, 'ub'):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must have .lower/.upper or .bounds.lb/.bounds.ub")
        dim = self.dim
        NP = self.NP
        budget = self.budget

        # Ensure at least 2 evaluations for initial population and one generation
        if budget < NP + 1:
            # Fallback to random search
            best_x = lb + (ub - lb) * np.random.rand(dim)
            best_y = func(best_x)
            evals = 1
            while evals < budget:
                x = lb + (ub - lb) * np.random.rand(dim)
                y = func(x)
                evals += 1
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
            return best_x, best_y

        # Initial population (uniform random)
        pop = lb + (ub - lb) * np.random.rand(NP, dim)
        y = np.empty(NP)
        for i in range(NP):
            y[i] = func(pop[i])
        evals = NP
        # Track best so far
        best_idx = np.argmin(y)
        best_x = pop[best_idx].copy()
        best_y = y[best_idx]

        # Initialize individual control parameters
        # F and CR for each individual (NP x 1)
        F = np.random.uniform(self.F_l, self.F_u, NP)
        CR = np.random.uniform(self.CR_l, self.CR_u, NP)

        # Main loop: one generation per iteration
        # Each generation uses NP evaluations (one per updated individual)
        while evals + NP <= budget:
            new_pop = pop.copy()
            new_y = y.copy()
            new_F = F.copy()
            new_CR = CR.copy()
            for i in range(NP):
                # Generate new parameters with probability tau1, tau2
                if np.random.rand() < self.tau1:
                    new_F[i] = np.random.uniform(self.F_l, self.F_u)
                if np.random.rand() < self.tau2:
                    new_CR[i] = np.random.uniform(self.CR_l, self.CR_u)
                # Choose three distinct random indices different from i
                candidates = list(range(NP))
                candidates.remove(i)
                r = np.random.choice(candidates, 3, replace=False)
                a, b, c = r[0], r[1], r[2]
                # Mutant vector
                mutant = pop[a] + new_F[i] * (pop[b] - pop[c])
                # Binomial crossover
                cross_points = np.random.rand(dim) < new_CR[i]
                # Ensure at least one coordinate from mutant
                if not np.any(cross_points):
                    cross_points[np.random.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                # Bound repair: replace out-of-bound coordinates with random in bounds
                oob_low = trial < lb
                oob_high = trial > ub
                if np.any(oob_low) or np.any(oob_high):
                    # For each coordinate that is out of bounds, pick a uniformly random value inside
                    random_repair = lb + (ub - lb) * np.random.rand(dim)
                    trial = np.where(oob_low | oob_high, random_repair, trial)
                # Evaluate trial
                trial_y = func(trial)
                evals += 1
                # Selection (minimization)
                if trial_y < y[i]:
                    new_pop[i] = trial
                    new_y[i] = trial_y
                    # Keep the new parameters (already set in new_F/new_CR)
                else:
                    # Keep old parameters
                    new_F[i] = F[i]
                    new_CR[i] = CR[i]
                # Update best if this trial improved
                if trial_y < best_y:
                    best_y = trial_y
                    best_x = trial.copy()
            # Replace population for next generation
            pop = new_pop
            y = new_y
            F = new_F
            CR = new_CR
            # Stop if budget almost exhausted (prevent overshoot)
            if evals + NP > budget:
                break

        # If any evaluations left after last full generation, use them for random perturbations
        while evals < budget:
            # Random perturbation from best solution
            x = best_x + 0.1 * (ub - lb) * np.random.randn(dim)
            # Clip to bounds
            x = np.clip(x, lb, ub)
            y = func(x)
            evals += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()

        return best_x, best_y
