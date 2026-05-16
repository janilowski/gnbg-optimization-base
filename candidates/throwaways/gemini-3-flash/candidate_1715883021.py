# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Real-Coded Genetic Algorithm (RCGA) utilizing Simulated Binary Crossover (SBX) and Polynomial Mutation for continuous optimization.
# Search state: Retains a population of candidate real-valued vectors, their objective fitness values, and the global best solution found.
# Candidate generation: Parents chosen via binary tournament selection generate offspring pairs using SBX and polynomial mutation operators.
# Selection and replacement: An elitist generational replacement strategy where the best parent is preserved and remaining offspring replace the old population.
# Adaptation: Distribution indices for SBX (eta_c) and mutation (eta_m) remain constant, maintaining consistent exploratory pressure.
# Exploration mechanisms: Polynomial mutation provides stochastic perturbations across all coordinate axes to maintain genetic diversity.
# Exploitation mechanisms: Tournament selection favors highly fit individuals, and SBX recombines successful parental values in close proximity.
# Boundary handling: Offspring coordinates are explicitly clipped to valid lower and upper domain boundaries.
# Budget strategy: Iterates through generational population batches while strictly verifying remaining evaluation budget limits.
# Closest known influences: Real-Coded Genetic Algorithms (Deb & Agrawal).
# Novelty or unusual aspects: Compact vectorized implementation of SBX and polynomial mutation bounded by dynamic budget constraints.
# Failure modes: Can exhibit slow fine-tuning convergence on highly multimodal or narrow valley landscapes compared to gradient-like methods.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 5, max(16, 2 * self.dim)))
        if self.pop_size % 2 != 0:
            self.pop_size += 1  # Keep even for pairing
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

        eta_c = 15.0  # SBX distribution index
        eta_m = 20.0  # Polynomial mutation index
        mut_prob = 1.0 / self.dim

        while self.eval_count < self.budget:
            next_pop = np.zeros_like(pop)
            next_fit = np.full(self.pop_size, float("inf"))

            # Elitism: preserve top individual
            best_idx = np.argmin(fitness)
            next_pop[0] = pop[best_idx].copy()
            next_fit[0] = fitness[best_idx]

            for i in range(1, self.pop_size, 2):
                if self.eval_count >= self.budget:
                    break

                # Binary tournament selection
                p1_idx = np.random.choice(self.pop_size, size=2, replace=False)
                p1 = pop[p1_idx[0]] if fitness[p1_idx[0]] < fitness[p1_idx[1]] else pop[p1_idx[1]]

                p2_idx = np.random.choice(self.pop_size, size=2, replace=False)
                p2 = pop[p2_idx[0]] if fitness[p2_idx[0]] < fitness[p2_idx[1]] else pop[p2_idx[1]]

                # SBX Crossover
                c1, c2 = p1.copy(), p2.copy()
                if np.random.rand() < 0.9:
                    u = np.random.rand(self.dim)
                    beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta_c + 1.0)), (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)))
                    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)

                # Polynomial Mutation
                for cand in (c1, c2):
                    mask = np.random.rand(self.dim) < mut_prob
                    if np.any(mask):
                        r = np.random.rand(self.dim)
                        delta = np.where(r <= 0.5, (2.0 * r) ** (1.0 / (eta_m + 1.0)) - 1.0, 1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0)))
                        cand[mask] += delta[mask] * domain_range[mask]

                # Clip and evaluate c1
                c1 = np.clip(c1, lb, ub)
                y1 = float(func(c1))
                self.eval_count += 1
                next_pop[i] = c1
                next_fit[i] = y1

                if y1 < best_y:
                    best_y = y1
                    best_x = c1.copy()

                if i + 1 < self.pop_size and self.eval_count < self.budget:
                    c2 = np.clip(c2, lb, ub)
                    y2 = float(func(c2))
                    self.eval_count += 1
                    next_pop[i + 1] = c2
                    next_fit[i + 1] = y2

                    if y2 < best_y:
                        best_y = y2
                        best_x = c2.copy()

            pop = next_pop
            fitness = next_fit

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
