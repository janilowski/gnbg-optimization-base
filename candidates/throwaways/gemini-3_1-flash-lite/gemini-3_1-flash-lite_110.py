# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Self-Adaptive Differential Evolution variant designed for black-box minimization.
# Search state: Maintains a current population of potential solutions (vectors) and their fitness values.
# Candidate generation: Uses a mutation operator (DE/rand/1) where a donor vector is created from the weighted difference of two individuals added to a third.
# Selection and replacement: Greedy selection; a candidate replaces its parent only if the new fitness is lower.
# Adaptation: Employs a self-adaptive control parameter for the scaling factor F, centered around 0.5.
# Exploration mechanisms: Population diversity is maintained by random sampling and the mutation vector scale.
# Exploitation mechanisms: The crossover operator mixes parent and mutant genes, refined by greedily keeping only improved solutions.
# Boundary handling: Clamping to provided lower/upper bounds.
# Budget strategy: Precisely exhausts the budget by filling the initial population and running cycles of evolution until the budget is depleted.
# Closest known influences: Classic Differential Evolution (Storn & Price).
# Novelty or unusual aspects: Minimalist implementation focusing on robustness via randomized parameter jittering.
# Failure modes: May struggle with highly deceptive (narrow global optima) landscapes or extreme dimensionality relative to population size.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(40, budget // 2)
        if self.pop_size < 4:
            self.pop_size = 4

    def __call__(self, func):
        # Determine bounds
        if hasattr(func, 'lower'):
            lb, ub = np.array(func.lower), np.array(func.upper)
        else:
            lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)

        # Initialization
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        eval_count = self.pop_size

        best_idx = np.argmin(fitness)
        best_x = pop[best_idx].copy()
        best_y = fitness[best_idx]

        # Optimization loop
        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                
                # Mutation: DE/rand/1 with jitter
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                # F is the scaling factor, randomized for exploration
                f = 0.5 + 0.3 * np.random.rand()
                mutant = a + f * (b - c)
                
                # Boundary clamping
                mutant = np.clip(mutant, lb, ub)
                
                # Crossover (Binomial)
                cross_points = np.random.rand(self.dim) < 0.5
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Selection
                score = func(trial)
                eval_count += 1
                
                if score < fitness[i]:
                    fitness[i] = score
                    pop[i] = trial
                    if score < best_y:
                        best_y = score
                        best_x = trial.copy()
                        
        return best_x, best_y
