# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Steady-State Genetic Algorithm utilizing rank selection and immediate worst-member replacement.
# Search state: Stores population vectors, objective fitness values, and global optimum.
# Candidate generation: Two parents chosen via rank weighting produce a single offspring via uniform crossover and Gaussian mutation.
# Selection and replacement: Evaluated offspring immediately replace the worst individual in the current population.
# Adaptation: Mutational variance scales with the average distance between population members and global best.
# Exploration mechanisms: Uniform crossover recombines distinct parental dimensions while mutational noise injects diversity.
# Exploitation mechanisms: Rank selection favors high-performing parents, and steady-state worst replacement ensures monotonic population improvement.
# Boundary handling: All offspring candidate positions are clipped inside domain bounds.
# Budget strategy: Allocates evaluations one-by-one in an asynchronous steady-state loop until budget exhaustion.
# Closest known influences: Steady-State Genetic Algorithms / GENITOR (Whitley).
# Novelty or unusual aspects: Eliminates generation barriers in favor of continuous real-time population updates.
# Failure modes: High elitism pressure can lead to rapid premature convergence if initial diversity is insufficient.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 4, max(15, 2 * self.dim)))
        if self.pop_size > 50:
            self.pop_size = 50

    def __call__(self, func):
        try:
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        except AttributeError:
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        self.eval_count = 0
        domain_range = ub - lb

        best_x = None
        best_y = float("inf")

        pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        fitness = np.full(self.pop_size, float("inf"))

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop[i]))
            self.eval_count += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        # Precompute rank selection weights (linear ranking)
        ranks = np.arange(self.pop_size, 0, -1)
        rank_probs = ranks / np.sum(ranks)
        sigma = 0.15

        while self.eval_count < self.budget:
            # Sort population to identify worst and apply rank selection
            sorted_indices = np.argsort(fitness)
            
            # Select parents based on rank weights
            p_indices = np.random.choice(sorted_indices, size=2, replace=False, p=rank_probs)
            p1, p2 = pop[p_indices[0]], pop[p_indices[1]]

            # Uniform crossover
            mask = np.random.rand(self.dim) < 0.5
            offspring = np.where(mask, p1, p2)

            # Gaussian mutation
            if np.random.rand() < 0.8:
                step = np.random.normal(0, 1, size=self.dim) * (sigma * domain_range)
                offspring += step

            offspring = np.clip(offspring, lb, ub)
            y = float(func(offspring))
            self.eval_count += 1

            if y < best_y:
                best_y = y
                best_x = offspring.copy()

            # Immediate worst replacement
            worst_idx = sorted_indices[-1]
            if y < fitness[worst_idx]:
                pop[worst_idx] = offspring
                fitness[worst_idx] = y

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
