# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An (mu + lambda) Evolution Strategy that dynamically scales mutation step sizes using Rechenberg's 1/5th success rule.
# Search state: Retains a population of mu parent vectors, their objective values, current mutation step size sigma, and the global best.
# Candidate generation: Offspring are created by adding isotropic Gaussian perturbations scaled by sigma and domain range to randomly selected parents.
# Selection and replacement: The top mu individuals from the combined pool of parents and offspring are selected to form the next generation.
# Adaptation: Mutation step size sigma increases if more than 20% of offspring improve upon their parents, and decreases otherwise.
# Exploration mechanisms: Isotropic Gaussian mutation across multiple parents maintains population diversity across the landscape.
# Exploitation mechanisms: Elitist (mu + lambda) selection ensures the best discovered solutions are never lost between generations.
# Boundary handling: All mutated offspring vectors are clipped to remain inside the feasible box domain before evaluation.
# Budget strategy: Evaluates batches of lambda offspring per generation, strictly stopping as soon as the total budget is reached.
# Closest known influences: Evolution Strategies (Rechenberg & Schwefel).
# Novelty or unusual aspects: Integrates global success rate tracking across a multi-parent population rather than a single trajectory.
# Failure modes: Can suffer from step size premature shrinkage if trapped on deceptive plateau landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0

        self.mu = 4
        self.lam = 16
        if self.lam > self.budget // 3:
            self.lam = max(2, self.budget // 3)
            self.mu = max(1, self.lam // 4)

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

        # Initialize parent population
        parents = np.random.uniform(lb, ub, size=(self.mu, self.dim))
        parents_y = np.zeros(self.mu)

        for i in range(self.mu):
            if self.eval_count >= self.budget:
                parents_y[i] = float("inf")
                continue
            y = float(func(parents[i]))
            self.eval_count += 1
            parents_y[i] = y
            if y < best_y:
                best_y = y
                best_x = parents[i].copy()

        sigma = 0.15

        while self.eval_count < self.budget:
            offspring = np.zeros((self.lam, self.dim))
            offspring_y = np.zeros(self.lam)
            success_count = 0

            for i in range(self.lam):
                if self.eval_count >= self.budget:
                    offspring_y[i] = float("inf")
                    continue

                parent_idx = np.random.randint(self.mu)
                step = np.random.normal(0, 1, size=self.dim) * (sigma * domain_range)
                cand = np.clip(parents[parent_idx] + step, lb, ub)

                y = float(func(cand))
                self.eval_count += 1
                offspring[i] = cand
                offspring_y[i] = y

                if y < parents_y[parent_idx]:
                    success_count += 1

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            if self.eval_count >= self.budget:
                break

            # 1/5th success rule adaptation
            success_rate = success_count / self.lam
            if success_rate > 0.2:
                sigma = min(1.2 * sigma, 0.5)
            elif success_rate < 0.2:
                sigma = max(0.85 * sigma, 1e-6)

            # (mu + lambda) selection
            pool_x = np.vstack((parents, offspring))
            pool_y = np.concatenate((parents_y, offspring_y))

            top_indices = np.argsort(pool_y)[:self.mu]
            parents = pool_x[top_indices]
            parents_y = pool_y[top_indices]

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
