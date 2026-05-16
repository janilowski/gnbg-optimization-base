# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Quantum-Behaved Differential Evolution algorithm hybridizing standard DE mutation with quantum exponential wave potential well updates.
# Search state: Retains active population positions, objective fitness values, and global optimum across generational iterations.
# Candidate generation: Half the swarm generates trial vectors via current-to-best DE mutation; half generates trials via quantum exponential well perturbations.
# Selection and replacement: Standard one-to-one parent replacement; updates global best optimum whenever trial solutions achieve superior objective fitness.
# Adaptation: Contraction-expansion coefficient alpha decreases linearly over iterations to tighten quantum potential wells around the global best.
# Exploration mechanisms: Difference vector scaling in both DE and quantum regimes maintains mutational reach across unvisited landscape valleys.
# Exploitation mechanisms: Quantum potential well updates centered on the global best optimum drive aggressive local basin convergence.
# Boundary handling: All trial vector positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Evaluates population trial vectors sequentially in generational iterations while checking remaining evaluation budget.
# Closest known influences: Quantum-Behaved Differential Evolution QDE / Hybrid Swarm.
# Novelty or unusual aspects: Directly scales quantum Laplacian wave potential well radius using population difference vectors.
# Failure modes: Can experience premature variance collapse if difference vectors contract faster than the local quadratic curvature.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 8, max(20, 2 * self.dim)))
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

        f, cr = 0.8, 0.8
        alpha_start, alpha_end = 1.0, 0.4
        half_pop = self.pop_size // 2

        while self.eval_count < self.budget:
            progress = self.eval_count / self.budget
            alpha = alpha_start - (alpha_start - alpha_end) * progress

            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                r1 = np.random.randint(self.pop_size)
                while r1 == i:
                    r1 = np.random.randint(self.pop_size)

                r2 = np.random.randint(self.pop_size)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(self.pop_size)

                if i < half_pop:
                    # DE/current-to-best/1/bin
                    v = pop[i] + f * (best_x - pop[i]) + f * (pop[r1] - pop[r2])
                    mask = np.random.rand(self.dim) <= cr
                    mask[np.random.randint(self.dim)] = True
                    trial = np.where(mask, v, pop[i])
                else:
                    # Quantum potential well update
                    u_val = np.random.uniform(0.001, 0.999, size=self.dim)
                    sign_flip = np.where(np.random.rand(self.dim) < 0.5, 1.0, -1.0)
                    step = alpha * np.abs(pop[r1] - pop[r2]) * np.log(1.0 / u_val) * sign_flip
                    trial = best_x + step

                trial = np.clip(trial, lb, ub)
                y = float(func(trial))
                self.eval_count += 1

                if y <= fitness[i]:
                    next_pop[i] = trial
                    next_fit[i] = y
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()
                else:
                    next_pop[i] = pop[i]
                    next_fit[i] = fitness[i]

            pop = next_pop
            fitness = next_fit

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
