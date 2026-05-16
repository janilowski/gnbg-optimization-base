# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Differential Evolution algorithm employing the DE/best/1/bin mutation strategy to guide search towards the global optimum.
# Search state: Retains population vectors, objective fitness values, and global best solution across generational cycles.
# Candidate generation: Generates trial vectors by adding a scaled difference between two random population members to the global best position.
# Selection and replacement: Evaluates trial vectors and replaces parent solutions if objective fitness improves or remains equal.
# Adaptation: Dithers the mutation scale factor F around a mean of 0.7 to maintain search diversity.
# Exploration mechanisms: Random selection of difference vectors from diverse population pairs allows exploration around the global optimum.
# Exploitation mechanisms: Anchoring mutation directly on the global best position drives aggressive convergence towards the leading basin.
# Boundary handling: Trial vectors are explicitly clipped inside valid domain boundaries.
# Budget strategy: Iterates through generational population loops while verifying remaining evaluation budget limits.
# Closest known influences: Differential Evolution DE/best/1/bin (Storn & Price).
# Novelty or unusual aspects: Employs random jitter on scale factor F per individual trial generation to avoid premature stagnation on ridges.
# Failure modes: Can get trapped in local minima if the global best anchor is positioned in a sub-optimal deceptive well.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 5, max(12, 3 * self.dim)))
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

        cr = 0.85

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # DE/best/1/bin
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2 = np.random.choice(candidates, size=2, replace=False)

                f_jitter = np.random.normal(0.7, 0.1)
                f_jitter = np.clip(f_jitter, 0.2, 1.0)

                mutant = best_x + f_jitter * (pop[r1] - pop[r2])

                # Binomial crossover
                cross_mask = np.random.rand(self.dim) < cr
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(self.dim)] = True

                trial = np.where(cross_mask, mutant, pop[i])
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
