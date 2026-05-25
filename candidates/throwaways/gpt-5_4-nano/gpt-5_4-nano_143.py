# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy using a population
# of candidate points and iterative “ask/evaluate/tell” style updates inside a
# fixed evaluation budget. It combines randomized exploration (multi-point
# sampling with a shrinking Gaussian radius) and exploitation via a simple
# best-so-far and local perturbations around the current best.
# Search state: Maintains a population of points (X) and their objective values
# (Y), plus the current global best (best_x, best_y) and the current sampling
# scale (sigma). Also tracks the remaining evaluation budget.
# Candidate generation: Each iteration draws new candidate points by adding
# Gaussian noise (scaled by sigma) to either the current best or random
# population members. A small portion of candidates uses uniform sampling to
# maintain diversity early and when progress stalls.
# Selection and replacement: After evaluating candidates, the algorithm updates
# the global best and forms the next population by taking the best candidates
# (elitist replacement). This keeps the population improving over time.
# Adaptation: sigma shrinks gradually with iterations to transition from
# exploration to exploitation. If no improvement is observed for several
# iterations, sigma is temporarily increased to escape local minima.
# Exploration mechanisms: Uniform random samples within bounds plus Gaussian
# perturbations with a relatively larger sigma early on and after stagnation.
# Exploitation mechanisms: Gaussian perturbations centered around the current
# best-so-far, with a smaller sigma as the algorithm progresses.
# Boundary handling: Any point that leaves the feasible bounds is clipped
# component-wise to stay within [lb, ub]. (Clipping is simple and robust.)
# Budget strategy: Computes the maximum number of function evaluations from
# the provided budget and never evaluates more than budget_total. Each iteration
# evaluates at most the remaining budget. Population size is chosen so that
# the algorithm fits the budget.
# Closest known influences: Inspired by simple evolution strategies / CMA-like
# heuristics at a lightweight scale: elitist selection, adaptive step size,
# and best-centered sampling.
# Novelty or unusual aspects: Uses an evaluation-budget-aware iteration plan,
# combines best-centered and population-centered Gaussian sampling, and includes
# a small uniform “reset” fraction for diversity without storing history.
# Failure modes: If the objective is extremely noisy or adversarial, the
# algorithm may stagnate; budget constraints can limit refinement. For very
# high-dimensional ill-conditioned landscapes, clipping may bias sampling.
# ALGORITHM_ANALYSIS_NOTE_END

from typing import Callable, Optional, Tuple

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")

    def __call__(self, func: Callable):
        # --------- Read bounds ---------
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)
        else:
            raise AttributeError("func must have (lower, upper) or bounds.lb/bounds.ub")

        if lb.shape != (self.dim,) or ub.shape != (self.dim,):
            lb = np.reshape(lb, (self.dim,))
            ub = np.reshape(ub, (self.dim,))
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        if np.any(ub < lb):
            raise ValueError("Invalid bounds: require ub >= lb for all dimensions")

        # Helper: clip to bounds
        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        # --------- Budget management ---------
        budget_total = self.budget
        evals = 0

        def eval_one(x):
            nonlocal evals
            # Ensure we never exceed the provided evaluation budget.
            if evals >= budget_total:
                return np.inf
            x = clip(np.asarray(x, dtype=float))
            y = func(x)
            evals += 1
            return float(y)

        # Handle trivial budget
        if budget_total == 1:
            # Evaluate a single point at the center.
            x0 = 0.5 * (lb + ub)
            y0 = eval_one(x0)
            return clip(x0), y0

        # --------- Hyperparameters (chosen to fit budget and dimension) ---------
        # Population size: smaller for high dimension to save budget.
        # Ensure at least 2 so we can do selection.
        pop = int(max(2, min(16, budget_total // 4)))
        pop = min(pop, budget_total)  # cannot exceed budget

        # Number of iterations (upper bounded by budget); each iteration evaluates "offspring".
        # We start with an initial population, then do further generations.
        # If budget is tight, offspring can become 1.
        # Plan: initial = pop (or remaining), then iterations with offspring = pop.
        remaining = budget_total
        # ---------- Initialize population ----------
        # Mix: best starting guess from midpoint plus random points.
        # Evaluate exactly pop points if possible.
        init_n = min(pop, remaining)
        X = np.empty((init_n, self.dim), dtype=float)
        Y = np.empty(init_n, dtype=float)

        x_mid = clip(0.5 * (lb + ub))
        if init_n >= 1:
            X[0] = x_mid
        # Fill rest with uniform random within bounds
        if init_n > 1:
            r = np.random.rand(init_n - 1, self.dim)
            X[1:] = lb + r * (ub - lb)

        for i in range(init_n):
            Y[i] = eval_one(X[i])

        # Best-so-far
        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])
        remaining = budget_total - evals

        # Initial sampling scale: proportional to average box width.
        box = (ub - lb)
        # Robust against zero-width dimensions
        avg_width = float(np.mean(box))
        # If all bounds are tight, sigma becomes 0 and search is deterministic around best.
        sigma = 0.25 * avg_width
        if sigma <= 0.0:
            sigma = 1e-12

        # Stagnation tracking for step-size adaptation
        no_improve = 0
        patience = max(5, min(25, budget_total // max(1, pop)))

        # How many more evaluations to spend after init?
        # Each generation evaluates offspring_n candidates.
        # Choose offspring_n so total fits budget.
        # We'll run until we can't afford another offspring set.
        while remaining > 0:
            # Determine offspring size: aim for pop, but never exceed remaining.
            offspring_n = min(pop, remaining)
            if offspring_n <= 0:
                break

            # Progress-based schedule: sigma shrinks over time.
            # We also slightly shrink at each step, but guard with max bounds.
            # Use fraction of budget spent to scale down.
            t = evals / max(1, budget_total)
            # shrink factor between ~1 and ~0.1
            shrink = (0.1 ** t)
            sigma_t = max(1e-12, sigma * shrink)

            # Stagnation escape: if stuck, temporarily boost sigma.
            if no_improve >= patience:
                sigma_t *= 2.0
                no_improve = 0

            # Diversity fraction: uniform random in early/mid; reduced later.
            # Use the same fraction for simplicity.
            diversity = 0.20 * (1.0 - t)
            diversity = float(np.clip(diversity, 0.05, 0.25))

            # Candidate generation:
            # - A subset from uniform (global exploration)
            # - The rest from Gaussian around the current best and some random parents
            K_uni = int(round(diversity * offspring_n))
            K_uni = min(K_uni, offspring_n)
            K_gau = offspring_n - K_uni

            X_new = np.empty((offspring_n, self.dim), dtype=float)

            # Uniform samples
            if K_uni > 0:
                ru = np.random.rand(K_uni, self.dim)
                X_new[:K_uni] = lb + ru * (ub - lb)

            # Gaussian samples around best and random population points
            # Mix modes: mostly around best (exploitation), some around random parents (diversity)
            if K_gau > 0:
                # Decide how many around best
                best_share = 0.7
                K_best = int(round(best_share * K_gau))
                K_best = min(K_best, K_gau)
                K_parent = K_gau - K_best

                # Around best
                if K_best > 0:
                    Z = np.random.randn(K_best, self.dim)
                    X_new[K_uni:K_uni + K_best] = best_x + sigma_t * Z

                # Around random parents (elitist parents from current X)
                if K_parent > 0:
                    # Pick random indices from current population.
                    idxs = np.random.randint(0, X.shape[0], size=K_parent)
                    centers = X[idxs]
                    Z = np.random.randn(K_parent, self.dim)
                    X_new[K_uni + K_best:K_uni + K_parent + K_best] = centers + sigma_t * Z

            # Evaluate candidates
            Y_new = np.empty(offspring_n, dtype=float)
            for i in range(offspring_n):
                Y_new[i] = eval_one(X_new[i])

            # Update remaining
            remaining = budget_total - evals
            if remaining < 0:
                remaining = 0

            # Selection: keep best individuals from combined pool (elitist replacement)
            # Merge current population with new candidates (but cap to pop size)
            X_comb = np.vstack([X, X_new])
            Y_comb = np.concatenate([Y, Y_new])

            # Sort by objective (minimization)
            order = np.argsort(Y_comb, kind="stable")
            keep = min(pop, X_comb.shape[0])
            order = order[:keep]
            X = X_comb[order]
            Y = Y_comb[order]

            # Global best update
            if float(np.min(Y_new)) < best_y:
                new_best_idx = int(np.argmin(Y_new))
                best_y_new = float(Y_new[new_best_idx])
                if best_y_new < best_y:
                    best_y = best_y_new
                    best_x = clip(X_new[new_best_idx]).copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            # Additional adaptation: shrink base sigma slowly when improving
            # (If sigma_t computed via shrink schedule is enough, this is mild.)
            # This is to make behavior stable even if shrink schedule is aggressive.
            if no_improve == 0:
                sigma *= 0.98
            else:
                # mild decay to keep progress
                sigma *= 0.995

            if evals >= budget_total:
                break

        return clip(best_x), best_y
