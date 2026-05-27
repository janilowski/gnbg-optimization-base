# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a Self-adaptive Differential Evolution (jDE) algorithm tailored for robust black-box minimization.
# Search state: Maintains a population of candidate vectors, their corresponding fitness values, and individual control parameters (mutation scale F and crossover rate CR).
# Candidate generation: Uses the 'DE/rand/1/bin' strategy, where a mutant vector is created by adding a scaled difference between two random population members to a third. Binomial crossover is then applied between the parent and mutant.
# Selection and replacement: Uses a greedy selection mechanism; a trial vector replaces the parent in the population only if its objective value is less than or equal to the parent's.
# Adaptation: Employs the jDE self-adaptation logic where mutation (F) and crossover (CR) parameters are updated stochastically for each individual, allowing successful configurations to persist and propagate.
# Exploration mechanisms: Driven by a relatively large initial population and the mutation factor F, which scales the search steps based on the current distribution of the population.
# Exploitation mechanisms: Binomial crossover allows for search along coordinate axes, while greedy selection focuses the search on promising regions of the landscape.
# Boundary handling: Trial vectors are strictly clipped to the specified lower and upper bounds before evaluation.
# Budget strategy: Monitored throughout the initialization and evolution phases. The algorithm terminates immediately once the function evaluation budget is exhausted.
# Closest known influences: The jDE algorithm by Brest et al. (2006) and the original Differential Evolution by Storn and Price.
# Novelty or unusual aspects: A lean, single-class implementation that automatically scales population size and handles diverse bound specifications without external configuration.
# Failure modes: May converge slowly on extremely high-dimensional, flat landscapes or get trapped in local optima on highly multi-modal functions if the population loses diversity too early.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # 1. Determine bounds
        if hasattr(func, "lower") and func.lower is not None:
            lb = np.asfarray(func.lower)
            ub = np.asfarray(func.upper)
        elif hasattr(func, "bounds"):
            lb = np.asfarray(func.bounds.lb)
            ub = np.asfarray(func.bounds.ub)
        else:
            # Fallback if no bounds are provided, though benchmark usually provides them
            lb = np.full(self.dim, -100.0)
            ub = np.full(self.dim, 100.0)

        # Ensure lb and ub are arrays of correct dimension
        if lb.ndim == 0: lb = np.full(self.dim, lb)
        if ub.ndim == 0: ub = np.full(self.dim, ub)

        # 2. Hyperparameters and Population Setup
        # Use a population size scaled by dimension, but constrained by budget and practical limits
        pop_size = min(self.budget, max(10, self.dim * 10))
        if pop_size < 4:
            pop_size = self.budget # Minimal possible population for tiny budgets

        # State initialization
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.full(pop_size, np.inf)
        
        # jDE parameters
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        tau1 = 0.1 # Probability to update F
        tau2 = 0.1 # Probability to update CR
        
        evals = 0
        best_x = None
        best_y = np.inf

        # 3. Initial Evaluation
        for i in range(pop_size):
            if evals >= self.budget:
                break
            y = func(pop[i])
            evals += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # 4. Evolution Loop
        while evals < self.budget:
            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # Self-adaptation of parameters F and CR
                curr_F = F[i]
                curr_CR = CR[i]
                if np.random.rand() < tau1:
                    curr_F = 0.1 + np.random.rand() * 0.9
                if np.random.rand() < tau2:
                    curr_CR = np.random.rand()

                # Differential Mutation (DE/rand/1)
                # Select 3 distinct indices other than i
                if pop_size >= 4:
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                    mutant = pop[r1] + curr_F * (pop[r2] - pop[r3])
                else:
                    # Fallback for extremely small populations
                    mutant = pop[i] + curr_F * np.random.standard_normal(self.dim)

                # Binomial Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where(
                    (np.random.rand(self.dim) < curr_CR) | (np.arange(self.dim) == j_rand),
                    mutant,
                    pop[i]
                )

                # Boundary Handling
                trial = np.clip(trial, lb, ub)

                # Evaluation
                y_trial = func(trial)
                evals += 1

                # Selection
                if y_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = y_trial
                    F[i] = curr_F
                    CR[i] = curr_CR
                    if y_trial < best_y:
                        best_y = y_trial
                        best_x = trial.copy()

        return best_x, best_y
