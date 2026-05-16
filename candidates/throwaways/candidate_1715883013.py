# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A continuous Biogeography-Based Optimization (BBO) algorithm modeling feature migration across candidate solution habitats.
# Search state: Retains a population of candidate habitats, their objective suitability indices, and global best solution.
# Candidate generation: Generates trial habitats via probabilistic feature migration between high suitability (low objective) and low suitability habitats.
# Selection and replacement: Trial habitats replace parent habitats if objective values improve or remain equal.
# Adaptation: Emigration and immigration probabilities scale dynamically based on relative suitability rankings within the current population.
# Exploration mechanisms: Stochastic Gaussian mutation applied to migrating features maintains habitat diversity.
# Exploitation mechanisms: High emigration rates from elite habitats rapidly disseminate successful coordinate values across the population.
# Boundary handling: All modified habitat coordinates are strictly clipped to valid domain bounds.
# Budget strategy: Iterates through habitats sequentially per generation while enforcing strict evaluation budget limits.
# Closest known influences: Biogeography-Based Optimization (Simon).
# Novelty or unusual aspects: Employs continuous Gaussian mutation instead of discrete replacement to better explore real-parameter domains.
# Failure modes: Can experience genetic drift in highly epistatic landscapes where coordinate mixing disrupts complex linkages.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 5, max(14, 2 * self.dim)))
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

        mut_prob = 0.1
        sigma = 0.05

        while self.eval_count < self.budget:
            # Rank population: 0 is best
            ranks = np.argsort(np.argsort(fitness))
            
            # Immigration rate lambda: increases with bad rank
            imm_rate = ranks / (self.pop_size - 1.0 + 1e-12)
            
            # Emigration rate mu: decreases with bad rank
            em_rate = 1.0 - imm_rate
            em_probs = em_rate / (np.sum(em_rate) + 1e-12)

            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                trial = pop[i].copy()

                for j in range(self.dim):
                    if np.random.rand() < imm_rate[i]:
                        # Select source habitat proportional to emigration rate
                        source_idx = np.random.choice(self.pop_size, p=em_probs)
                        trial[j] = pop[source_idx, j]

                    if np.random.rand() < mut_prob:
                        step = np.random.normal(0, 1) * (sigma * domain_range[j])
                        trial[j] += step

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
