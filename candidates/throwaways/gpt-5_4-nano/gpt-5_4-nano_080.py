# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a robust black-box minimizer using a lightweight
# derivative-free evolutionary strategy with restarts. It maintains a small
# population of candidate points and iteratively improves them using Gaussian
# mutations around the current best solutions.
#
# Search state: Tracks a population of points, their fitness values, the
# current best point/value, and a step size (sigma) that controls mutation
# magnitude. Also tracks how many evaluations have been used.
#
# Candidate generation: Each iteration generates offspring by sampling
# Gaussian noise scaled by sigma around selected parents. Offspring are
# clipped to provided bounds.
#
# Selection and replacement: Uses elitist (µ+λ) selection: the next population
# is formed from the best individuals among parents and offspring. This
# ensures monotonic non-increasing best-so-far objective value.
#
# Adaptation: Sigma adapts based on success (improvement) rate of recent
# offspring. If improvements are frequent, sigma is decreased slightly
# (more exploitation); if not, sigma is increased slightly (more exploration).
#
# Exploration mechanisms: Random restarts are triggered when progress stalls,
# reinitializing part of the population uniformly over the search domain.
#
# Exploitation mechanisms: Gaussian mutations centered at the current best
# and strong parents focus search locally. Elitism retains the best points.
#
# Boundary handling: Any generated candidate is clipped to the feasible
# box defined by func.lower/upper or func.bounds.lb/ub.
#
# Budget strategy: Never evaluates more than the given budget. It stops once
# the budget is exhausted; evaluation calls are counted precisely.
#
# Closest known influences: Inspired by simple evolution strategies / CMA-like
# behavior (but with far less overhead): elitist selection, self-adaptive step
# size, and restarts.
#
# Novelty or unusual aspects: Uses a compact µ+λ ES with success-rate driven
# sigma adaptation plus a stall-based partial restart, designed to be
# dimension-agnostic and evaluation-budget aware.
#
# Failure modes: If the objective landscape is extremely deceptive or the
# budget is extremely small, sigma adaptation and restarts may not help.
# Additionally, if bounds are very tight or degenerate, clipping can reduce
# effective diversity.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from func ----
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective must provide bounds via lower/upper or bounds.lb/bounds.ub")

        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))
        lb = lb.reshape(dim)
        ub = ub.reshape(dim)
        # Ensure valid bounds
        if np.any(ub < lb):
            raise ValueError("Invalid bounds: ub must be >= lb elementwise")

        # If budget is 0: no evaluations; return a clipped random guess.
        if budget <= 0:
            x0 = np.clip(np.random.rand(dim) * (ub - lb) + lb, lb, ub)
            return x0, float("inf")

        # ---- Helper functions ----
        eval_count = 0
        best_x = None
        best_y = float("inf")

        def eval_one(x):
            nonlocal eval_count, best_x, best_y
            y = float(func(x))
            eval_count += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        # ---- Hyperparameters (chosen to be robust across dims) ----
        # Population size: small enough to preserve budget for iterations.
        # Use at least 4 points when possible.
        mu = min(8, budget)  # parents size
        mu = max(4, mu)
        # Offspring per iteration
        lam = min(12, max(4, budget - mu))
        # We'll create at most lam offspring each iteration, but may reduce near budget end.
        max_iters = 10_000  # actual stopping by budget

        # Initial sigma based on domain scale; avoid 0 division.
        scale = ub - lb
        # If some dimensions have zero range, sigma in those dims stays 0 after clipping.
        domain = np.max(scale) if np.max(scale) > 0 else 1.0
        sigma = 0.3 * domain / (dim ** 0.5)

        # Track recent improvement to adapt sigma and decide restart.
        recent_improvements = 0
        recent_window = 10  # check success rate over last N offspring

        # Initialize population uniformly in bounds
        pop = np.empty((mu, dim), dtype=float)
        fitness = np.empty(mu, dtype=float)
        for i in range(mu):
            x = clip(np.random.rand(dim) * (ub - lb) + lb)
            pop[i] = x
            fitness[i] = eval_one(pop[i])
            if eval_count >= budget:
                return best_x, best_y

        # Current best is already tracked by eval_one.
        # Main loop: µ+λ with adaptation and restarts.
        it = 0
        stall_iters = 0
        best_y_prev = best_y

        while eval_count < budget and it < max_iters:
            it += 1

            # ---- Selection: pick parents from current population (tournament) ----
            # Use a bias toward better individuals.
            # Tournament size depends weakly on dim.
            tsize = 2 if dim < 10 else 3

            def tournament_select():
                idxs = np.random.randint(0, mu, size=tsize)
                # Minimize fitness
                return idxs[np.argmin(fitness[idxs])]

            parents_idx = np.array([tournament_select() for _ in range(lam)], dtype=int)

            # ---- Candidate generation: Gaussian mutations around selected parents ----
            # We'll generate offspring and evaluate until budget runs out.
            offspring = np.empty((lam, dim), dtype=float)
            off_fit = np.empty(lam, dtype=float)

            # Decrease sigma slightly if best improved recently; boost if not.
            # (More stable behavior than raw jump sizes.)
            # This factor is modest and always clipped by restart logic.
            sigma_min = 1e-12
            sigma_max = max(1.0, domain)  # absolute cap (in domain units)

            improved_in_this_iter = False
            offspring_evaluated = 0

            for k in range(lam):
                if eval_count >= budget:
                    break

                p = parents_idx[k]
                parent = pop[p]

                # Mutation: isotropic Gaussian with occasional center at best.
                # This mixes exploration and exploitation.
                center = parent
                if np.random.rand() < 0.35:
                    center = best_x if best_x is not None else parent

                # Sample noise; use sigma and small per-dim variation for robustness.
                noise = np.random.randn(dim) * sigma
                x = center + noise

                # Boundary handling
                x = clip(x)

                offspring[k] = x
                y = eval_one(x)
                off_fit[k] = y
                offspring_evaluated += 1

                if y < best_y_prev:
                    improved_in_this_iter = True
                    recent_improvements += 1

            if offspring_evaluated == 0:
                break

            # Reduce arrays to evaluated portion
            offspring = offspring[:offspring_evaluated]
            off_fit = off_fit[:offspring_evaluated]

            # ---- Selection and replacement: µ+λ elitist ----
            # Combine parents and offspring then take best mu.
            combined = np.vstack([pop, offspring])
            combined_fit = np.concatenate([fitness, off_fit])

            # Take top mu smallest fitness
            order = np.argsort(combined_fit)[:mu]
            pop = combined[order]
            fitness = combined_fit[order]

            # ---- Adaptation: sigma based on recent success rate ----
            # Success defined as "offspring better than previous best".
            # Update and slightly adapt sigma.
            window_success = recent_improvements
            # Normalize by how many offspring were generated/evaluated recently.
            # Since we don't store history, approximate based on this iteration.
            denom = max(1, offspring_evaluated)
            success_rate = window_success / denom

            # Decay sigma if success; otherwise increase.
            if improved_in_this_iter:
                # More exploitation after improvement.
                sigma *= (0.86 + 0.1 * (1.0 - success_rate))
                stall_iters = 0
            else:
                # More exploration if stuck.
                sigma *= (1.08 + 0.15 * min(1.0, 1.0 / (1.0 + success_rate)))
                stall_iters += 1

            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # Reset recent improvements window periodically
            # (keeps adaptation bounded and avoids lock-in).
            recent_improvements = 0
            best_y_prev = best_y

            # ---- Exploration mechanism: partial restart on stall ----
            # If no improvement for some iterations, reinitialize part of population.
            if stall_iters >= 4 and eval_count < budget:
                # Reinitialize about 40% (at least 1) individuals uniformly.
                num_restart = max(1, int(0.4 * mu))
                worst_idx = np.argsort(fitness)[::-1][:num_restart]  # largest fitness

                for idx in worst_idx:
                    if eval_count >= budget:
                        break
                    x = clip(np.random.rand(dim) * (ub - lb) + lb)
                    pop[idx] = x
                    fitness[idx] = eval_one(x)

                # After restart, widen sigma to recover exploration.
                sigma = min(sigma_max, sigma * 1.25)
                stall_iters = 0

        # If best_x never set (shouldn't happen if mu>=1 and budget>0)
        if best_x is None:
            # Evaluate nothing or all evals failed? Fallback: random within bounds.
            best_x = clip(np.random.rand(dim) * (ub - lb) + lb)
            best_y = float("inf")

        return best_x, best_y
