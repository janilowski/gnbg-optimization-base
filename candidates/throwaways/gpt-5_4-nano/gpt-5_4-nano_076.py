# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization routine using an adaptive
# population of candidate points and a simple Gaussian random search around the
# current best. It is robust across dimensions and respects the evaluation budget.
# Search state: Maintains a small population of points, their objective values,
# the current best solution (best_x, best_y), and a scalar step size (sigma).
# Candidate generation: Each iteration samples candidates either by perturbing the
# best point with Gaussian noise (exploitation) or by drawing uniformly within bounds
# (exploration), then clips them to stay inside bounds.
# Selection and replacement: Evaluates candidates, keeps the best solutions in the
# population (elitist replacement), and updates the global best with any improvement.
# Adaptation: Updates sigma multiplicatively based on recent success rate using a
# simple 1/5-style rule (if many improvements occur, increase exploitation strength;
# otherwise reduce step size).
# Exploration mechanisms: Uses occasional uniform sampling to escape local minima.
# Exploitation mechanisms: Primarily relies on Gaussian perturbations centered at the
# current best.
# Boundary handling: All candidate points are clipped to [lower, upper] derived from
# func.lower/func.upper or func.bounds.lb/ub.
# Budget strategy: Never evaluates more than the provided budget. The algorithm
# converts the budget into a maximum number of objective calls, and computes how many
# candidates to evaluate per iteration to fit within the remaining budget.
# Closest known influences: Inspired by (mu+lambda) evolution strategies and the classic
# 1/5 success rule for step-size adaptation, but implemented in a lightweight,
# budget-aware way.
# Novelty or unusual aspects: Uses a dynamic mixture schedule between uniform exploration
# and best-centered Gaussian search, while adapting sigma based on observed improvements.
# Failure modes: If the objective is extremely noisy or the budget is very small, progress
# may be limited; sigma adaptation can be conservative and rely more on uniform exploration.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _get_bounds(self, func):
        # Prefer func.lower/func.upper; otherwise use func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            b = getattr(func, "bounds")
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)

        if lb.shape == ():
            lb = np.full(self.dim, float(lb))
        if ub.shape == ():
            ub = np.full(self.dim, float(ub))

        lb = np.reshape(lb, (self.dim,))
        ub = np.reshape(ub, (self.dim,))
        if not np.all(ub >= lb):
            raise ValueError("Invalid bounds: require ub >= lb elementwise.")
        return lb, ub

    def _clip(self, x, lb, ub):
        return np.minimum(np.maximum(x, lb), ub)

    def __call__(self, func):
        lb, ub = self._get_bounds(func)
        dim = self.dim
        n_eval = 0

        def eval_one(x):
            nonlocal n_eval
            # Enforce budget strictly.
            if n_eval >= self.budget:
                # If this triggers, the calling logic is wrong; keep safe fallback.
                return np.inf
            y = float(func(x))
            n_eval += 1
            return y

        # Handle trivial/degenerate budgets.
        if self.budget <= 0:
            # No evaluations allowed; return something deterministic within bounds.
            x0 = (lb + ub) / 2.0
            return x0, float("inf")

        rng = np.random  # harness sets np.random seed

        # Initialization: evaluate a few points uniformly + mid point.
        # Use an evaluation count that fits in the budget.
        max_init = min(10, self.budget)
        init_points = max(1, max_init - 1)  # leave room for midpoint
        init_X = rng.uniform(low=lb, high=ub, size=(init_points, dim))
        init_Y = []
        for i in range(init_points):
            init_Y.append(eval_one(init_X[i]))
            if n_eval >= self.budget:
                break

        # Always evaluate midpoint if budget allows.
        if n_eval < self.budget:
            x_mid = (lb + ub) / 2.0
            y_mid = eval_one(x_mid)
            init_X = np.vstack([init_X, x_mid])
            init_Y.append(y_mid)

        init_Y = np.asarray(init_Y, dtype=float)
        best_idx = int(np.argmin(init_Y))
        best_x = np.asarray(init_X[best_idx], dtype=float)
        best_y = float(init_Y[best_idx])

        # Population size for elitist replacement.
        # Keep it small for speed, but enough for selection pressure.
        mu = max(2, min(8, self.budget))  # number of elites to keep
        mu = min(mu, self.budget - n_eval + mu)  # safe-ish
        mu = max(2, min(mu, 12))

        # Sigma initialization: proportional to the box size.
        box = np.maximum(ub - lb, 1e-12)
        sigma = 0.3 * box  # per-dimension scale works better than a scalar

        # Keep a population of elites. Start from evaluated initial points.
        # If fewer than mu points were evaluated, pad by perturbing best.
        X_pop = np.asarray(init_X, dtype=float)
        Y_pop = np.asarray(init_Y, dtype=float)

        # Ensure population doesn't exceed budget.
        while X_pop.shape[0] < min(mu, self.budget) and n_eval < self.budget:
            cand = self._clip(best_x + rng.normal(size=dim) * sigma, lb, ub)
            y = eval_one(cand)
            X_pop = np.vstack([X_pop, cand])
            Y_pop = np.append(Y_pop, y)

        # Sort and retain elites.
        order = np.argsort(Y_pop)
        X_pop = X_pop[order]
        Y_pop = Y_pop[order]
        pop_keep = min(mu, X_pop.shape[0])
        X_pop = X_pop[:pop_keep]
        Y_pop = Y_pop[:pop_keep]

        # Main loop: decide how many candidates to evaluate per iteration
        # based on remaining budget.
        remaining = self.budget - n_eval
        # If no remaining budget, return best so far.
        if remaining <= 0:
            return best_x, best_y

        # Mixture schedule: more exploration early, less later.
        # We'll reduce uniform probability as budget gets consumed.
        # Also cap candidates per iteration to keep logic simple.
        max_cands = min(24, max(4, self.budget // 4))

        # Success tracking for sigma adaptation.
        # Use a rolling window over the last few evaluations batch outcomes.
        window = 0
        successes = 0

        while n_eval < self.budget:
            remaining = self.budget - n_eval
            # Candidates per iteration: fit within remaining budget.
            batch = min(max_cands, remaining)
            if batch <= 0:
                break

            frac_explore = 0.35 * (remaining / max(1, self.budget))  # decreasing over time
            frac_explore = float(np.clip(frac_explore, 0.05, 0.35))

            n_explore = int(np.round(batch * frac_explore))
            n_exploit = batch - n_explore

            candidates = []

            # Exploration: uniform sampling in bounds.
            if n_explore > 0:
                X_u = rng.uniform(low=lb, high=ub, size=(n_explore, dim))
                candidates.append(X_u)

            # Exploitation: Gaussian around current best (with mild additional noise).
            if n_exploit > 0:
                Z = rng.normal(size=(n_exploit, dim))
                # Use per-dimension sigma to better adapt to anisotropic bounds.
                X_g = best_x + Z * sigma
                X_g = self._clip(X_g, lb, ub)
                candidates.append(X_g)

            if candidates:
                C = np.vstack(candidates)
            else:
                # Shouldn't happen, but keep safe.
                C = rng.uniform(low=lb, high=ub, size=(batch, dim))

            # Evaluate candidates and record improvements.
            Y_new = np.empty((C.shape[0],), dtype=float)
            improved_count = 0
            for i in range(C.shape[0]):
                y = eval_one(C[i])
                Y_new[i] = y
                if y < best_y:
                    improved_count += 1
                    best_y = y
                    best_x = np.asarray(C[i], dtype=float)

                if n_eval >= self.budget:
                    # If budget ran out mid-batch, stop evaluating.
                    # Trim remaining Y_new.
                    Y_new = Y_new[: i + 1]
                    C = C[: i + 1]
                    break

            # Update rolling success metrics for sigma adaptation.
            # "Success" defined as producing a point better than current best.
            # This is a bit stringent but works well for minimization.
            window = min(20, window + int(C.shape[0]))
            successes = min(20, successes + improved_count)

            # Elitist replacement: combine current population with evaluated candidates.
            # Keep the best pop_keep points.
            X_comb = np.vstack([X_pop, C]) if C.shape[0] > 0 else X_pop
            Y_comb = np.concatenate([Y_pop, Y_new]) if Y_new.size > 0 else Y_pop
            order = np.argsort(Y_comb)
            X_pop = X_comb[order][:pop_keep]
            Y_pop = Y_comb[order][:pop_keep]

            # Sigma adaptation based on observed success proportion.
            # 1/5-like: if many successes, slightly increase sigma to explore;
            # else decrease sigma to exploit.
            # Use robust scaling with per-dimension box.
            # Avoid division by zero.
            denom = max(1, window)
            success_rate = successes / denom

            # Target around 0.2 success rate.
            target = 0.2
            if success_rate > target:
                # Increase step size moderately to broaden search.
                sigma = sigma * 1.08
            else:
                # Decrease step size to focus exploitation.
                sigma = sigma / 1.08

            # Prevent sigma from becoming too large or too small relative to box.
            sigma_min = box * 1e-6
            sigma_max = box * 0.8
            sigma = np.minimum(np.maximum(sigma, sigma_min), sigma_max)

            # Reset rolling window occasionally to make adaptation responsive.
            if window >= 12:
                window = 0
                successes = 0

        return best_x, best_y
