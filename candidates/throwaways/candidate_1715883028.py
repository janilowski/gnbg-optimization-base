# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: A Dual-Population Genetic Algorithm running concurrent exploratory and exploitative sub-populations with periodic migration.
# Search state: Retains two independent sub-populations of candidate vectors, their objective values, generation counters, and global best.
# Candidate generation: Sub-pop 1 uses high mutation for exploration; Sub-pop 2 uses SBX crossover and low mutation for exploitation.
# Selection and replacement: Generational tournament selection within each sub-population, with elite preservation in Sub-pop 2.
# Adaptation: Periodically exchanges elite individuals between sub-populations every 10 generations to cross-pollinate search traits.
# Exploration mechanisms: High mutation rates in Sub-pop 1 maintain global diversity and prevent premature convergence.
# Exploitation mechanisms: Low mutation and SBX crossover in Sub-pop 2 rapidly refine leading candidate positions.
# Boundary handling: All offspring candidate positions are explicitly clipped inside domain bounds.
# Budget strategy: Alternates generational steps between sub-populations while strictly monitoring remaining evaluation budget.
# Closest known influences: Parallel Genetic Algorithms / Island Models (Cantú-Paz).
# Novelty or unusual aspects: Combines heterogeneous search strategies (exploration vs exploitation) in a synchronous dual-island architecture.
# Failure modes: Can divide evaluation budget sub-optimally if the landscape is purely unimodal or purely rugged.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)
        self.eval_count = 0
        self.sub_pop = int(min(self.budget // 10, max(12, 2 * self.dim)))
        if self.sub_pop > 30:
            self.sub_pop = 30

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

        pop1 = np.random.uniform(lb, ub, size=(self.sub_pop, self.dim))
        fit1 = np.full(self.sub_pop, float("inf"))

        pop2 = np.random.uniform(lb, ub, size=(self.sub_pop, self.dim))
        fit2 = np.full(self.sub_pop, float("inf"))

        # Initialize pop1
        for i in range(self.sub_pop):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop1[i]))
            self.eval_count += 1
            fit1[i] = y
            if y < best_y:
                best_y = y
                best_x = pop1[i].copy()

        # Initialize pop2
        for i in range(self.sub_pop):
            if self.eval_count >= self.budget:
                break
            y = float(func(pop2[i]))
            self.eval_count += 1
            fit2[i] = y
            if y < best_y:
                best_y = y
                best_x = pop2[i].copy()

        gen = 0
        eta_c = 15.0

        while self.eval_count < self.budget:
            gen += 1

            # --- Evolve Pop 1 (Exploration: High Mutation) ---
            next_pop1 = np.zeros_like(pop1)
            next_fit1 = np.full(self.sub_pop, float("inf"))
            
            best1_idx = np.argmin(fit1)
            next_pop1[0] = pop1[best1_idx].copy()
            next_fit1[0] = fit1[best1_idx]

            for i in range(1, self.sub_pop):
                if self.eval_count >= self.budget:
                    break
                t = np.random.choice(self.sub_pop, size=2, replace=False)
                p = pop1[t[0]] if fit1[t[0]] < fit1[t[1]] else pop1[t[1]]

                # High Gaussian mutation
                step = np.random.normal(0, 1, size=self.dim) * (0.25 * domain_range)
                cand = np.clip(p + step, lb, ub)
                
                y = float(func(cand))
                self.eval_count += 1
                next_pop1[i] = cand
                next_fit1[i] = y

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            pop1 = next_pop1
            fit1 = next_fit1

            # --- Evolve Pop 2 (Exploitation: SBX + Low Mutation) ---
            if self.eval_count >= self.budget:
                break

            next_pop2 = np.zeros_like(pop2)
            next_fit2 = np.full(self.sub_pop, float("inf"))

            best2_idx = np.argmin(fit2)
            next_pop2[0] = pop2[best2_idx].copy()
            next_fit2[0] = fit2[best2_idx]

            for i in range(1, self.sub_pop):
                if self.eval_count >= self.budget:
                    break
                t1 = np.random.choice(self.sub_pop, size=2, replace=False)
                p1 = pop2[t1[0]] if fit2[t1[0]] < fit2[t1[1]] else pop2[t1[1]]

                t2 = np.random.choice(self.sub_pop, size=2, replace=False)
                p2 = pop2[t2[0]] if fit2[t2[0]] < fit2[t2[1]] else pop2[t2[1]]

                # SBX
                cand = p1.copy()
                if np.random.rand() < 0.9:
                    u = np.random.rand(self.dim)
                    beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta_c + 1.0)), (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)))
                    cand = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)

                # Low mutation
                mask = np.random.rand(self.dim) < (1.0 / self.dim)
                if np.any(mask):
                    step = np.random.normal(0, 1, size=self.dim) * (0.02 * domain_range)
                    cand[mask] += step[mask]

                cand = np.clip(cand, lb, ub)
                y = float(func(cand))
                self.eval_count += 1
                next_pop2[i] = cand
                next_fit2[i] = y

                if y < best_y:
                    best_y = y
                    best_x = cand.copy()

            pop2 = next_pop2
            fit2 = next_fit2

            # --- Migration every 10 generations ---
            if gen % 10 == 0:
                best1_idx, worst1_idx = np.argmin(fit1), np.argmax(fit1)
                best2_idx, worst2_idx = np.argmin(fit2), np.argmax(fit2)

                # Swap
                pop1[worst1_idx] = pop2[best2_idx].copy()
                fit1[worst1_idx] = fit2[best2_idx]

                pop2[worst2_idx] = pop1[best1_idx].copy()
                fit2[worst2_idx] = fit1[best1_idx]

        if best_x is None:
            best_x = np.random.uniform(lb, ub, size=self.dim)

        return best_x, best_y
