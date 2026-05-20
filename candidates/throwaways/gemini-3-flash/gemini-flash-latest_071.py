# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a Self-Adaptive Differential Evolution (jDE) algorithm. This is a population-based metaheuristic that automatically tunes its mutation and crossover parameters during the search process.
# Search state: A population of candidate solution vectors, their associated objective function values, and per-individual mutation (F) and crossover (CR) parameters.
# Candidate generation: Uses the 'rand/1/bin' strategy. A trial vector is created by adding a scaled difference between two random population members to a third random member, followed by binomial crossover with the parent.
# Selection and replacement: A greedy selection mechanism where the trial vector replaces the parent in the population if its objective value is less than or equal to the parent's.
# Adaptation: Self-adaptation of parameters F and CR. For each trial vector generation, there is a small probability (0.1) that these parameters are updated, allowing the algorithm to discover and propagate effective settings.
# Exploration mechanisms: Initial uniform sampling across the search space and the stochastic nature of the differential mutation.
# Exploitation mechanisms: Greedy selection pressure and the inherent property of DE where the search step size naturally decreases as the population converges.
# Boundary handling: Trial vectors are clipped to the lower and upper bounds defined by the problem.
# Budget strategy: The algorithm tracks evaluations and terminates immediately once the budget is exhausted. An initial fallback handles cases where the budget is too small to form a minimum population.
# Closest known influences: jDE (Brest et al., 2006).
# Novelty or unusual aspects: A lean implementation of jDE designed for robustness across various dimensions within a single-class structure.
# Failure modes: May struggle with extremely low budgets in very high dimensions or high-frequency multimodal landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Target a population size between 4 and 100 based on dimensionality.
        self.pop_size = int(max(4, min(100, 10 * dim)))

    def __call__(self, func):
        # Extract bounds from the function object
        if hasattr(func, "bounds") and hasattr(func.bounds, "lb"):
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        elif hasattr(func, "lower"):
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        else:
            # Fallback if no bounds are detected
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)

        eval_count = 0
        best_y = float('inf')
        best_x = None

        # Fallback for extremely small budgets or populations
        if self.budget < self.pop_size or self.pop_size < 4:
            for _ in range(self.budget):
                x = np.random.uniform(lb, ub, self.dim)
                y = func(x)
                eval_count += 1
                if y < best_y:
                    best_y, best_x = y, x.copy()
            return (best_x, best_y) if best_x is not None else (np.random.uniform(lb, ub, self.dim), 0.0)

        # Initialize population and parameters
        X = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.full(self.pop_size, float('inf'))
        F = np.full(self.pop_size, 0.5)    # Scaling factor
        CR = np.full(self.pop_size, 0.9)   # Crossover rate

        # Initial evaluations
        for i in range(self.pop_size):
            if eval_count >= self.budget:
                break
            fitness[i] = func(X[i])
            eval_count += 1
            if fitness[i] < best_y:
                best_y = fitness[i]
                best_x = X[i].copy()

        # Evolution loop
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break

                # jDE Parameter Adaptation
                # Probability tau1 = 0.1 to update F, tau2 = 0.1 to update CR
                f_i = F[i]
                cr_i = CR[i]
                if np.random.rand() < 0.1:
                    f_i = 0.1 + np.random.rand() * 0.9
                if np.random.rand() < 0.1:
                    cr_i = np.random.rand()

                # Mutation (rand/1 strategy)
                # Select three distinct indices different from i
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                v = X[r1] + f_i * (X[r2] - X[r3])
                
                # Binomial Crossover
                j_rand = np.random.randint(self.dim)
                mask = np.random.rand(self.dim) < cr_i
                mask[j_rand] = True # Ensure at least one component comes from the mutant
                trial = np.where(mask, v, X[i])
                
                # Boundary Handling: Clip trial vector to allowed search space
                trial = np.clip(trial, lb, ub)
                
                # Evaluation
                f_trial = func(trial)
                eval_count += 1
                
                # Selection: Greedy replacement
                if f_trial <= fitness[i]:
                    X[i] = trial
                    fitness[i] = f_trial
                    F[i] = f_i
                    CR[i] = cr_i
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()

        return best_x, best_y
