# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A robust Self-Adaptive Differential Evolution (jDE) variant utilizing a current-to-pbest mutation strategy.
# Search state: A population of candidate solutions, their corresponding fitness values, and individual-specific control parameters (F and CR).
# Candidate generation: Mutants are generated using a "current-to-pbest/1" strategy, which balances the influence of the best-performing individuals with random differential vectors. Binomial crossover is then applied.
# Selection and replacement: Standard one-to-one greedy selection where a trial vector replaces its parent in the population if it achieves a lower or equal objective value.
# Adaptation: Control parameters F (mutation scale) and CR (crossover probability) are adapted stochastically for each individual, allowing the algorithm to self-tune the search intensity and direction.
# Exploration mechanisms: Differential vectors derived from random members of the population maintain diversity and provide varied search directions.
# Exploitation mechanisms: The mutation strategy incorporates a "p-best" vector (randomly selected from the top 15% of the population) to guide search towards promising regions.
# Boundary handling: Simple clipping of candidate vectors to the problem's defined lower and upper bounds.
# Budget strategy: The population size is scaled by dimension but capped relative to the total budget to ensure a sufficient number of evolutionary generations.
# Closest known influences: Storn & Price (Differential Evolution), Brest et al. (jDE), Zhang & Sanderson (JADE).
# Novelty or unusual aspects: Highly compact implementation of adaptive parameters within a single-class structure, optimized for robustness across varying dimensions.
# Failure modes: May converge slowly on extremely flat landscapes or get trapped in strong local optima on highly deceptive functions if the population size is too small.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.count = 0

    def __call__(self, func):
        # Extract bounds from func
        if hasattr(func, 'bounds') and func.bounds is not None:
            lb = np.asarray(func.bounds.lb)
            ub = np.asarray(func.bounds.ub)
        elif hasattr(func, 'lower') and func.lower is not None:
            lb = np.asarray(func.lower)
            ub = np.asarray(func.upper)
        else:
            # Fallback if no bounds are provided
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        # Population size logic: proportional to dim but capped by budget
        pop_size = int(np.clip(10 * self.dim, 10, 100))
        if pop_size * 2 > self.budget:
            pop_size = max(4, self.budget // 2)

        # Initialize population
        pop = lb + np.random.rand(pop_size, self.dim) * (ub - lb)
        fitness = np.zeros(pop_size)
        
        # Adaptive parameters: F (scale) and CR (crossover rate)
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        
        best_x = None
        best_y = float('inf')

        # Initial evaluation
        for i in range(pop_size):
            if self.count >= self.budget:
                break
            val = func(pop[i])
            self.count += 1
            fitness[i] = val
            if val < best_y:
                best_y = val
                best_x = pop[i].copy()

        # Evolution loop
        while self.count < self.budget:
            # Sort indices for p-best selection
            indices = np.argsort(fitness)
            p_size = max(1, int(0.15 * pop_size))
            p_best_indices = indices[:p_size]

            for i in range(pop_size):
                if self.count >= self.budget:
                    break

                # Update F and CR with probability 0.1 (jDE adaptation)
                if np.random.rand() < 0.1:
                    F[i] = 0.1 + 0.9 * np.random.rand()
                if np.random.rand() < 0.1:
                    CR[i] = np.random.rand()

                # Mutation: current-to-pbest/1
                pbest = pop[np.random.choice(p_best_indices)]
                
                # Pick two distinct random indices different from i
                r_indices = np.random.choice([idx for idx in range(pop_size) if idx != i], 2, replace=False)
                r1, r2 = pop[r_indices[0]], pop[r_indices[1]]
                
                mutant = pop[i] + F[i] * (pbest - pop[i]) + F[i] * (r1 - r2)
                
                # Crossover (Binomial)
                cross_mask = np.random.rand(self.dim) <= CR[i]
                # Ensure at least one dimension is changed
                cross_mask[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_mask, mutant, pop[i])
                
                # Boundary handling: clip
                trial = np.clip(trial, lb, ub)
                
                # Selection
                f_trial = func(trial)
                self.count += 1
                
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_y:
                        best_y = f_trial
                        best_x = trial.copy()

        return best_x, best_y
