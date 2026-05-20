# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Robust Adaptive Differential Evolution (RADE) variant. It employs a dynamic population size and stochastic parameter adaptation to maintain a balance between exploration and exploitation.
# Search state: A population of candidate solutions (vectors) and their associated objective values, along with the current best solution found.
# Candidate generation: New candidates are generated using a 'rand/1/bin' strategy where a mutant vector is created from three random individuals and then crossed with the target individual.
# Selection and replacement: Standard DE greedy selection is used; a trial vector replaces its parent in the population only if it yields a lower or equal objective value.
# Adaptation: Mutation scale (F) and crossover probability (CR) are sampled from uniform distributions for each mutation step, allowing the search to automatically transition between different search step sizes and mixing intensities.
# Exploration mechanisms: Random selection of vectors for mutation (rand/1) and a jittered mutation factor F prevent premature convergence.
# Exploitation mechanisms: Binomial crossover ensures that improvements in specific dimensions are preserved, while greedy selection drives the population toward local minima.
# Boundary handling: Candidate vectors are strictly clipped to the defined lower and upper bounds before evaluation.
# Budget strategy: The algorithm tracks every evaluation and terminates as soon as the budget is exhausted, even within a generation loop.
# Closest known influences: Classic Differential Evolution (Storn & Price) and JADE-style parameter sampling.
# Novelty or unusual aspects: Compact implementation of stochastic parameter adaptation without requiring a historical memory or complex success-tracking.
# Failure modes: High-dimensional, extremely narrow ridges or highly deceptive landscapes may cause slow convergence if the budget is insufficient relative to the dimension.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    """
    An adaptive Differential Evolution algorithm designed for black-box minimization.
    It handles boundary constraints and adheres strictly to a provided evaluation budget.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        # Population size adaptive to dimension, capped for efficiency and budget constraints
        self.pop_size = min(max(10, 8 * dim), 100)
        self.eval_count = 0
        self.best_x = None
        self.best_y = float('inf')

    def __call__(self, func):
        # Extract bounds from the provided function object
        if hasattr(func, 'lower') and hasattr(func, 'upper'):
            lb = np.array(func.lower)
            ub = np.array(func.upper)
        elif hasattr(func, 'bounds'):
            lb = np.array(func.bounds.lb)
            ub = np.array(func.bounds.ub)
        else:
            # Fallback for safety, though GNBG benchmarks typically provide the above
            lb = np.zeros(self.dim)
            ub = np.ones(self.dim)

        # Initialize population
        pop = lb + np.random.rand(self.pop_size, self.dim) * (ub - lb)
        fitness = np.zeros(self.pop_size)

        # Evaluate initial population
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                return self.best_x, self.best_y
            
            val = func(pop[i])
            self.eval_count += 1
            fitness[i] = val
            
            if val < self.best_y:
                self.best_y = val
                self.best_x = np.copy(pop[i])

        # Main evolution loop
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Stochastic parameter selection (Adaptation)
                f_scale = np.random.uniform(0.4, 1.0)
                cr_rate = np.random.uniform(0.1, 0.9)

                # Mutation: rand/1 strategy
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + f_scale * (pop[b] - pop[c])

                # Boundary handling: Clipping
                mutant = np.clip(mutant, lb, ub)

                # Crossover: Binomial
                cross_points = np.random.rand(self.dim) < cr_rate
                # Ensure at least one dimension is changed
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_val = func(trial)
                self.eval_count += 1

                if trial_val <= fitness[i]:
                    fitness[i] = trial_val
                    pop[i] = trial
                    if trial_val < self.best_y:
                        self.best_y = trial_val
                        self.best_x = np.copy(trial)

        return self.best_x, self.best_y

# The harness typically creates the Algorithm instance and calls it with the objective.
# The numpy random seed is managed externally by the benchmark suite.
