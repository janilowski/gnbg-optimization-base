# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Differential Evolution algorithm using target-to-best mutation and exponential crossover to optimize continuous domains.
# Search state: Retains population solution vectors, objective fitness values, and global best solution across generations.
# Candidate generation: Generates trial vectors by shifting current positions towards the global best and adding a difference vector, followed by exponential crossover.
# Selection and replacement: Evaluated trial vectors replace parent vectors if objective fitness improves or remains equal.
# Adaptation: Employs a dithered mutation scale factor F between 0.5 and 0.9 each generation.
# Exploration mechanisms: Difference vectors from random pairs combined with exponential crossover maintain search space exploration.
# Exploitation mechanisms: Attraction vector towards the global best solution drives steady convergence towards the elite basin.
# Boundary handling: All trial positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Generates population trial solutions sequentially while strictly checking remaining evaluation budget.
# Closest known influences: Differential Evolution DE/target-to-best/1/exp (Storn & Price).
# Novelty or unusual aspects: Integrates exponential crossover with circular buffer dimension indexing in a compact vectorized loop.
# Failure modes: Can experience genetic drift along uncrossed dimensions if crossover rate CR is set too low on highly non-separable functions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 5, max(14, 2 * self.dim)))
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

        cr = 0.80

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # DE/target-to-best/1
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(candidates, size=2, replace=False)

                f = np.random.uniform(0.5, 0.9)
                mutant = pop[i] + f * (best_x - pop[i]) + f * (pop[r1] - pop[r2])

                # Exponential crossover
                trial = pop[i].copy()
                start_idx = np.random.randint(self.dim)
                curr_idx = start_idx
                l = 0

                while l < self.dim:
                    trial[curr_idx] = mutant[curr_idx]
                    curr_idx = (curr_idx + 1) % self.dim
                    l += 1
                    if np.random.rand() >= cr:
                        break

                trial = np.clip(trial, lb, ub)
                y = float(func(trial))
                self.eval_count += 1

                if y <= fitness[i]:
                    fitness[i] = y
                    pop[i] = trial
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
