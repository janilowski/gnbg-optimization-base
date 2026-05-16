# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: An Age-Layered Population Structure (ALPS) Genetic Algorithm restricting competition across age tiers to prevent premature convergence.
# Search state: Retains three distinct age-layered sub-populations (young, middle, veteran), individual age counters, objective values, and global best.
# Candidate generation: Parents selected within corresponding or immediately younger age layers produce offspring via arithmetic crossover and mutation.
# Selection and replacement: Offspring age increments with generational cycles; older individuals exceeding layer thresholds migrate to senior layers.
# Adaptation: Periodically flushes and re-seeds the bottom age layer (Layer 0) with fresh random samples to continually inject novel genetic material.
# Exploration mechanisms: Layered isolation prevents older dominant elites from immediately wiping out younger exploratory trajectories.
# Exploitation mechanisms: The senior veteran layer (Layer 2) preserves elite genetic material indefinitely and refines it via tournament selection.
# Boundary handling: All offspring candidate positions are explicitly clipped inside valid domain boundaries.
# Budget strategy: Iterates across the three age layers sequentially per generation while enforcing strict evaluation budget caps.
# Closest known influences: Age-Layered Population Structure ALPS (Hornby).
# Novelty or unusual aspects: Simplified continuous age migration across fixed tier sizes with continuous Gaussian mutation.
# Failure modes: Can experience higher evaluation overhead maintaining multiple layers on extremely simple unimodal landscapes.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.layer_size = int(min(self.budget // 15, max(8, self.dim)))
        if self.layer_size > 20:
            self.layer_size = 20

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

        n_layers = 3
        # 3 layers: 0=young, 1=middle, 2=veteran
        pops = [np.random.uniform(lb, ub, size=(self.layer_size, self.dim)) for _ in range(n_layers)]
        fits = [np.full(self.layer_size, float("inf")) for _ in range(n_layers)]
        ages = [np.zeros(self.layer_size, dtype=int) for _ in range(n_layers)]

        # Initial evaluation
        for l in range(n_layers):
            for i in range(self.layer_size):
                if self.eval_count >= self.budget:
                    break
                y = float(func(pops[l][i]))
                self.eval_count += 1
                fits[l][i] = y
                if y < best_y:
                    best_y = y
                    best_x = pops[l][i].copy()

        max_ages = [10, 30, 999999]
        sigma = 0.15
        gen = 0

        while self.eval_count < self.budget:
            gen += 1

            next_pops = [np.zeros_like(pops[l]) for l in range(n_layers)]
            next_fits = [np.full(self.layer_size, float("inf")) for l in range(n_layers)]
            next_ages = [np.zeros(self.layer_size, dtype=int) for l in range(n_layers)]

            for l in range(n_layers):
                if self.eval_count >= self.budget:
                    break

                # Elitism: keep best in layer
                best_idx = np.argmin(fits[l])
                next_pops[l][0] = pops[l][best_idx].copy()
                next_fits[l][0] = fits[l][best_idx]
                next_ages[l][0] = ages[l][best_idx] + 1

                for i in range(1, self.layer_size):
                    if self.eval_count >= self.budget:
                        break

                    # Parent 1 from layer l
                    t1 = np.random.choice(self.layer_size, size=2, replace=False)
                    p1_idx = t1[0] if fits[l][t1[0]] < fits[l][t1[1]] else t1[1]
                    p1 = pops[l][p1_idx]
                    age1 = ages[l][p1_idx]

                    # Parent 2 from layer l or l-1
                    mating_layer = l
                    if l > 0 and np.random.rand() < 0.5:
                        mating_layer = l - 1
                    
                    t2 = np.random.choice(self.layer_size, size=2, replace=False)
                    p2_idx = t2[0] if fits[mating_layer][t2[0]] < fits[mating_layer][t2[1]] else t2[1]
                    p2 = pops[mating_layer][p2_idx]
                    age2 = ages[mating_layer][p2_idx]

                    # Arithmetic crossover
                    alpha = np.random.uniform(0.1, 0.9)
                    offspring = alpha * p1 + (1.0 - alpha) * p2

                    # Mutation
                    if np.random.rand() < 0.5:
                        step = np.random.normal(0, 1, size=self.dim) * (sigma * domain_range)
                        offspring += step

                    offspring = np.clip(offspring, lb, ub)
                    y = float(func(offspring))
                    self.eval_count += 1

                    next_pops[l][i] = offspring
                    next_fits[l][i] = y
                    next_ages[l][i] = max(age1, age2) + 1

                    if y < best_y:
                        best_y = y
                        best_x = offspring.copy()

            pops = next_pops
            fits = next_fits
            ages = next_ages

            # Inter-layer migration
            for l in range(n_layers - 1):
                for i in range(self.layer_size):
                    if ages[l][i] > max_ages[l]:
                        # Compete with worst in layer l+1
                        worst_next_idx = np.argmax(fits[l+1])
                        if fits[l][i] < fits[l+1][worst_next_idx]:
                            pops[l+1][worst_next_idx] = pops[l][i].copy()
                            fits[l+1][worst_next_idx] = fits[l][i]
                            ages[l+1][worst_next_idx] = ages[l][i]

                        # Re-seed slot in layer 0 with random
                        if l == 0:
                            pops[0][i] = np.random.uniform(lb, ub, size=self.dim)
                            if self.eval_count < self.budget:
                                y = float(func(pops[0][i]))
                                self.eval_count += 1
                                fits[0][i] = y
                                if y < best_y:
                                    best_y = y
                                    best_x = pops[0][i].copy()
                            ages[0][i] = 0

            # Periodically flush layer 0 entirely
            if gen % 10 == 0 and self.eval_count < self.budget:
                for i in range(self.layer_size):
                    if self.eval_count >= self.budget:
                        break
                    pops[0][i] = np.random.uniform(lb, ub, size=self.dim)
                    y = float(func(pops[0][i]))
                    self.eval_count += 1
                    fits[0][i] = y
                    ages[0][i] = 0
                    if y < best_y:
                        best_y = y
                        best_x = pops[0][i].copy()

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
