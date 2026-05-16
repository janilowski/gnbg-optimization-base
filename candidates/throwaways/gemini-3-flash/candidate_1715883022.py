# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Genetic Algorithm employing Blend Crossover (BLX-alpha) and Gaussian mutation to optimize real-parameter spaces.
# Search state: Retains a population of candidate vectors, their objective fitness values, and the global optimum found across generations.
# Candidate generation: Offspring are created by uniformly sampling the expanded interval between paired parents (BLX-alpha) and applying Gaussian mutations.
# Selection and replacement: Top two elite members are preserved each generation, while tournament selection populates the remaining generational offspring slots.
# Adaptation: Mutation step size decreases linearly over the generation cycles to transition from global dispersion to local convergence.
# Exploration mechanisms: BLX-alpha crossover expands search bounds beyond parental positions by alpha proportional distance.
# Exploitation mechanisms: Tournament selection and elitism ensure elite genetic material dominates future recombinations.
# Boundary handling: All offspring candidate positions are clipped inside domain bounds.
# Budget strategy: Evaluates population members sequentially in generational cycles until the exact evaluation budget is exhausted.
# Closest known influences: Genetic Algorithm with BLX-alpha Crossover (Eshelman & Schaffer).
# Novelty or unusual aspects: Integrates linear mutational step size decay with uniform blend expansion in a compact generational loop.
# Failure modes: Can experience premature convergence if parental positions collapse too rapidly in early generations.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 5, max(16, 2 * self.dim)))
        if self.pop_size > 60:
            self.pop_size = 60

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

        alpha = 0.5
        mut_prob = 0.15
        sigma_start = 0.2
        sigma_end = 0.01

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            sigma = sigma_start - (sigma_start - sigma_end) * progress

            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            # Elitism: preserve top 2
            sorted_idx = np.argsort(fitness)
            n_elites = min(2, self.pop_size)
            for e in range(n_elites):
                next_pop[e] = pop[sorted_idx[e]].copy()
                next_fit[e] = fitness[sorted_idx[e]]

            for i in range(n_elites, self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Tournament selection
                t1 = np.random.choice(self.pop_size, size=3, replace=False)
                p1 = pop[t1[np.argmin(fitness[t1])]]

                t2 = np.random.choice(self.pop_size, size=3, replace=False)
                p2 = pop[t2[np.argmin(fitness[t2])]]

                # BLX-alpha crossover
                min_p = np.minimum(p1, p2)
                max_p = np.maximum(p1, p2)
                diff = max_p - min_p
                low_bound = min_p - alpha * diff
                high_bound = max_p + alpha * diff
                
                offspring = np.random.uniform(low_bound, high_bound)

                # Gaussian mutation
                mask = np.random.rand(self.dim) < mut_prob
                if np.any(mask):
                    step = np.random.normal(0, 1, size=self.dim) * (sigma * domain_range)
                    offspring[mask] += step[mask]

                offspring = np.clip(offspring, lb, ub)
                y = float(func(offspring))
                self.eval_count += 1

                next_pop[i] = offspring
                next_fit[i] = y

                if y < best_y:
                    best_y = y
                    best_x = offspring.copy()

            pop = next_pop
            fitness = next_fit

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
