# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This algorithm implements a Success-History-based Adaptive Differential Evolution (SHADE) variant optimized for black-box minimization. It features adaptive control parameters (F and CR) driven by historical success, dynamic population sizing scaled to the budget and dimensionality, an external archive to maintain population diversity, and robust boundary correction.
# Search state: The search state consists of a population of candidate vectors, their corresponding objective function values, an external archive of historical parent vectors, and a memory buffer of historical success parameters (M_F and M_CR).
# Candidate generation: Offspring are generated using a "current-to-pbest/1" mutation scheme, followed by binomial crossover. The mutation vector blends the current vector, a target chosen from the top-p fraction of the population, and differential vectors from the current population and the external archive.
# Selection and replacement: Offspring replace their parents if their objective value is equal to or better than the parent's. When a parent is replaced, its vector is added to the external archive, and its F and CR values are recorded as successful parameters.
# Adaptation: The memory parameters (M_F and M_CR) are updated at the end of each generation using a weighted Lehmer mean of the successful parameters, where weights are proportional to the objective function improvement.
# Exploration mechanisms: Exploration is promoted by utilizing an external archive of discarded solutions for mutation, applying Cauchy-distributed scale factors (F), and re-initializing the worst half of the population if the fitness variance drops below a convergence threshold (to escape local optima).
# Exploitation mechanisms: Exploitation is directed by the "current-to-pbest/1" strategy, which biases mutations towards the best-performing individuals in the current generation.
# Boundary handling: A midpoint boundary correction is used: if a mutated component violates the bounds, it is projected halfway between its parent value and the violated bound, preserving directional search information.
# Budget strategy: Population size is scaled dynamically to the budget and dimension. Evaluation counts are monitored globally via a strict evaluation wrapper to ensure the budget is never exceeded.
# Closest known influences: The SHADE algorithm (Tanabe & Fukunaga, 2013) and the JADE algorithm (Zhang & Sanderson, 2009).
# Novelty or unusual aspects: Dynamic budget-based fallback to random search for low-budget scenarios, and an automatic diversity recovery trigger that re-randomizes converging subpopulations.
# Failure modes: The algorithm may adapt parameters slowly on highly rugged, non-continuous, or extremely noisy landscapes where the correlation between successful steps and parameter choices is weak.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Retrieve bounds
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.atleast_1d(func.lower)
            ub = np.atleast_1d(func.upper)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.atleast_1d(func.bounds.lb)
            ub = np.atleast_1d(func.bounds.ub)
        else:
            lb = np.full(self.dim, -5.0)
            ub = np.full(self.dim, 5.0)

        if len(lb) == 1 and self.dim > 1:
            lb = np.full(self.dim, lb[0])
        if len(ub) == 1 and self.dim > 1:
            ub = np.full(self.dim, ub[0])

        eval_count = 0
        best_x = None
        best_y = float("inf")

        def evaluate(x):
            nonlocal eval_count, best_x, best_y
            if eval_count >= self.budget:
                return best_y
            val = func(x)
            eval_count += 1
            if val < best_y or best_x is None:
                best_y = val
                best_x = np.copy(x)
            return val

        # If budget is extremely small, fall back to pure random search
        if self.budget < 12:
            for _ in range(self.budget):
                x = lb + np.random.rand(self.dim) * (ub - lb)
                evaluate(x)
            return best_x, best_y

        # Determine population size based on budget and dimension
        N = int(np.clip(self.budget // 12, 6, 15 * self.dim))
        
        # Initialize population
        pop = lb + np.random.rand(N, self.dim) * (ub - lb)
        pop_fit = np.zeros(N)
        for i in range(N):
            if eval_count >= self.budget:
                break
            pop_fit[i] = evaluate(pop[i])

        # Archive for diversity preservation
        archive = []
        archive_max_size = N

        # SHADE historical memory setup
        H = 10
        M_CR = np.full(H, 0.5)
        M_F = np.full(H, 0.5)
        memory_idx = 0

        while eval_count < self.budget:
            # Re-initialize worst solutions if the population converges prematurely
            if np.std(pop_fit) < 1e-10:
                sorted_idx = np.argsort(pop_fit)
                for idx in sorted_idx[N // 2 :]:
                    if eval_count >= self.budget:
                        break
                    pop[idx] = lb + np.random.rand(self.dim) * (ub - lb)
                    pop_fit[idx] = evaluate(pop[idx])

            success_CR = []
            success_F = []
            fitness_diff = []

            for i in range(N):
                if eval_count >= self.budget:
                    break

                # Choose historical parameters
                r_idx = np.random.randint(0, H)
                cr = np.random.normal(M_CR[r_idx], 0.1)
                cr = np.clip(cr, 0.0, 1.0)

                # Cauchy distribution for F
                f = M_F[r_idx] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                while f <= 0.0:
                    f = M_F[r_idx] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
                if f > 1.0:
                    f = 1.0

                # Current-to-pbest/1/bin mutation
                pbest_num = max(2, int(0.15 * N))
                sorted_pop_idx = np.argsort(pop_fit)
                pbest_idx = np.random.choice(sorted_pop_idx[:pbest_num])
                x_pbest = pop[pbest_idx]

                # Select r1 from population
                r1_candidates = [idx for idx in range(N) if idx != i]
                r1 = np.random.choice(r1_candidates)

                # Select r2 from population + archive
                r2_pool = [pop[idx] for idx in range(N) if idx != i and idx != r1] + archive
                r2_idx = np.random.randint(0, len(r2_pool))
                x_r2 = r2_pool[r2_idx]

                # Mutation
                v = pop[i] + f * (x_pbest - pop[i]) + f * (pop[r1] - x_r2)

                # Crossover
                j_rand = np.random.randint(0, self.dim)
                u = np.empty(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < cr or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]

                # Boundary correction (midpoint projection)
                for j in range(self.dim):
                    if u[j] < lb[j]:
                        u[j] = (lb[j] + pop[i][j]) / 2.0
                    elif u[j] > ub[j]:
                        u[j] = (ub[j] + pop[i][j]) / 2.0

                # Evaluation
                u_fit = evaluate(u)

                # Selection
                if u_fit <= pop_fit[i]:
                    # Update archive
                    archive.append(np.copy(pop[i]))
                    if len(archive) > archive_max_size:
                        archive.pop(np.random.randint(0, len(archive)))

                    # Track success metrics
                    df = pop_fit[i] - u_fit
                    if df > 0:
                        success_CR.append(cr)
                        success_F.append(f)
                        fitness_diff.append(df)

                    pop[i] = u
                    pop_fit[i] = u_fit

            # Update historical memories if improvements were made
            if len(fitness_diff) > 0:
                sum_df = sum(fitness_diff)
                weights = [df / sum_df for df in fitness_diff]

                # Lehmer mean of F
                num_F = sum(w * (f**2) for w, f in zip(weights, success_F))
                den_F = sum(w * f for w, f in zip(weights, success_F))
                M_F[memory_idx] = num_F / den_F if den_F > 0 else M_F[memory_idx]

                # Weighted mean of CR
                M_CR[memory_idx] = sum(w * cr for w, cr in zip(weights, success_CR))

                memory_idx = (memory_idx + 1) % H

        return best_x, best_y
