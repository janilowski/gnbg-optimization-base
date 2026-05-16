# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Self-Adaptive Differential Evolution (jDE) algorithm where mutation scale factors and crossover rates co-evolve alongside population vectors.
# Search state: Retains population solution vectors, objective values, global best, and individual F and CR parameters for each population member.
# Candidate generation: Proposes trial vectors via DE/rand/1/bin mutation and crossover using individual-specific mutated F and CR parameters.
# Selection and replacement: Parent vectors and their associated F and CR parameters are replaced if trial vectors achieve equal or superior fitness.
# Adaptation: Each population member mutates its F and CR parameters stochastically prior to trial generation; successful parameter combinations inherit to the next generation.
# Exploration mechanisms: Stochastic parameter resets (tau1 and tau2 probabilities) prevent population-wide parameter stagnation and maintain genetic diversity.
# Exploitation mechanisms: Highly fit individuals carrying optimal local F and CR parameters rapidly propagate their successful search steps.
# Boundary handling: All generated trial vectors are clipped inside the valid lower and upper domain boundaries.
# Budget strategy: Iterates through generational population loops while ensuring strict compliance with evaluation budget caps.
# Closest known influences: Self-Adaptive Differential Evolution jDE (Brest et al.).
# Novelty or unusual aspects: Embeds parameter self-adaptation directly into individual genome tracking to eliminate external hyperparameter tuning.
# Failure modes: Can experience slow early progress if initial random parameter assignments are suboptimal.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.pop_size = int(min(self.budget // 5, max(15, 3 * self.dim)))
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

        F = np.full(self.pop_size, 0.5)
        CR = np.full(self.pop_size, 0.9)

        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop[i]))
            self.eval_count += 1
            fitness[i] = y
            if y < best_y:
                best_y = y
                best_x = pop[i].copy()

        tau1 = 0.1
        tau2 = 0.1

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break

                # Mutate parameters
                new_F = F[i]
                if np.random.rand() < tau1:
                    new_F = 0.1 + np.random.rand() * 0.9

                new_CR = CR[i]
                if np.random.rand() < tau2:
                    new_CR = np.random.rand()

                # Choose 3 distinct random members not equal to i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

                mutant = pop[r1] + new_F * (pop[r2] - pop[r3])

                # Binomial Crossover
                cross_mask = np.random.rand(self.dim) < new_CR
                if not np.any(cross_mask):
                    cross_mask[np.random.randint(self.dim)] = True

                trial = np.where(cross_mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                y = float(func(trial))
                self.eval_count += 1

                if y <= fitness[i]:
                    fitness[i] = y
                    pop[i] = trial
                    F[i] = new_F
                    CR[i] = new_CR
                    if y < best_y:
                        best_y = y
                        best_x = trial.copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
