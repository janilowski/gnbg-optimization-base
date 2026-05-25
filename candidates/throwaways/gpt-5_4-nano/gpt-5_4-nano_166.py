# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# (a randomized evolution strategy with restarts and local refinement). It is
# designed for arbitrary dimensionality and only relies on NumPy.
#
# Search state: The algorithm maintains a population of candidate points, their
# objective values, and a current "best" solution found so far. It also tracks
# a step size (sigma) that controls how far new candidates are sampled.
#
# Candidate generation: Each iteration samples a small population around a
# weighted mean of the best individuals using Gaussian perturbations. It
# optionally mixes in a directional mutation based on the difference between
# the current best and the population mean to encourage progress.
#
# Selection and replacement: From the combined pool (old + offspring), it keeps
# the top individuals (lowest objective values) to form the next population.
# The global best is updated whenever a new point improves the best known value.
#
# Adaptation: The step size sigma adapts using a success rule: if the best in the
# new offspring improves over the previous best, sigma is slightly increased or
# decreased depending on improvement trends (kept simple and robust). When progress
# stalls, sigma is reduced and a restart is triggered.
#
# Exploration mechanisms: Restarts and sampling with relatively large sigma
# encourage exploration. Periodic restart triggers when no improvement is
# observed for a while.
#
# Exploitation mechanisms: Weighted recombination (mean of top individuals) plus
# local refinement around the best candidate encourages exploitation.
#
# Boundary handling: All candidates are clipped to the provided search bounds.
#
# Budget strategy: The algorithm strictly counts objective evaluations and stops
# before exceeding the provided budget. It uses an initial sampling phase and then
# iterates with offspring batches; each iteration consumes exactly the number of
# new points evaluated.
#
# Closest known influences: Inspired by (μ, λ)-ES / CMA-like weighted recombination
# ideas, plus randomized restarts and simple step-size adaptation.
#
# Novelty or unusual aspects: The directional mutation term adds a cheap heuristic
# to bias exploration toward promising directions without requiring gradients.
#
# Failure modes: If the objective is extremely noisy or highly deceptive, step
# size adaptation may oscillate; restarts mitigate this but cannot guarantee
# success under all conditions.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        if d <= 0:
            raise ValueError("dim must be positive")
        budget = max(0, int(self.budget))
        if budget == 0:
            # No evaluations allowed; return a feasible point deterministically (midpoint).
            lb, ub = self._read_bounds(func)
            x = (lb + ub) / 2.0
            return x, float("inf")

        lb, ub = self._read_bounds(func)
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.shape[0] != d or ub.shape[0] != d:
            raise ValueError("Bounds dimension mismatch with dim")

        # Ensure bounds are ordered.
        low = np.minimum(lb, ub)
        high = np.maximum(lb, ub)

        rng = np.random

        evals = 0

        # Wrapper with strict budget enforcement.
        def eval_x(x):
            nonlocal evals
            if evals >= budget:
                # This should not happen if logic is correct.
                return float("inf")
            y = float(func(np.clip(x, low, high)))
            evals += 1
            return y

        # Initial population size.
        # Choose small but effective values; ensure at least 2.
        pop = int(np.clip(8 + d // 2, 8, 24))
        pop = min(pop, budget) if budget > 0 else pop

        # Random initialization uniformly within bounds.
        X = rng.uniform(low, high, size=(pop, d))
        y = np.array([eval_x(X[i]) for i in range(pop)], dtype=float)

        best_idx = int(np.argmin(y))
        best_x = X[best_idx].copy()
        best_y = float(y[best_idx])

        # Parameters for ES-like search.
        # Start sigma as a fraction of the average range.
        ranges = np.maximum(1e-12, high - low)
        sigma = 0.25 * float(np.mean(ranges))
        sigma = max(sigma, 1e-12)

        # Selection: keep top k.
        k = max(2, pop // 2)

        # Restarts and stagnation handling.
        stagnation = 0
        stagnation_limit = max(10, 2 * d)  # adaptive-ish
        restart_cooldown = 0

        # Directional mutation weight.
        dir_weight = 0.2

        # Main loop: generate offspring in batches without exceeding budget.
        while evals < budget:
            # Remaining evaluations.
            rem = budget - evals

            # Offspring count per loop.
            # Use at least 1; keep batch reasonably small for responsiveness.
            lam = int(np.clip(pop, 4, 28))
            lam = min(lam, rem)
            if lam <= 0:
                break

            # Weighted recombination mean from current population:
            # select best k individuals and weight them by inverse rank.
            order = np.argsort(y)
            top = X[order[:k]]
            # Rank-based weights (normalized).
            ranks = np.arange(k, dtype=float)
            w = 1.0 / (1.0 + ranks)
            w /= np.sum(w)
            mean = np.sum(top * w[:, None], axis=0)

            # Directional heuristic: move in the direction from mean to best.
            # (Encourages exploiting the best basin.)
            direction = best_x - mean

            # Sample offspring: mean + N(0,1)*sigma with a directional component.
            Z = rng.standard_normal(size=(lam, d))
            # Direction is normalized to avoid exploding steps.
            dn = float(np.linalg.norm(direction))
            if dn > 0:
                dir_unit = direction / dn
            else:
                dir_unit = np.zeros(d, dtype=float)

            # Directional term scaled by sigma and a random coefficient.
            coeff = rng.standard_normal(lam) * dir_weight
            offspring = mean + sigma * Z + (sigma * coeff)[:, None] * dir_unit[None, :]

            # Clip and evaluate.
            offspring = np.clip(offspring, low, high)

            y_off = np.empty(lam, dtype=float)
            for i in range(lam):
                if evals >= budget:
                    y_off[i] = float("inf")
                    continue
                y_off[i] = eval_x(offspring[i])

            # Combine selection (μ, λ style): merge current X with offspring.
            # Keep best pop individuals.
            X_cand = np.vstack([X, offspring])
            y_cand = np.concatenate([y, y_off])

            # Remove any inf due to budget overrun safeguard.
            finite_mask = np.isfinite(y_cand)
            X_cand = X_cand[finite_mask]
            y_cand = y_cand[finite_mask]

            if X_cand.shape[0] == 0:
                break

            ord2 = np.argsort(y_cand)
            X = X_cand[ord2[:pop]]
            y = y_cand[ord2[:pop]]

            # Update global best.
            cur_best_idx = int(np.argmin(y))
            cur_best_x = X[cur_best_idx]
            cur_best_y = float(y[cur_best_idx])

            improved = cur_best_y < best_y
            if improved:
                best_y = cur_best_y
                best_x = cur_best_x.copy()
                stagnation = 0
            else:
                stagnation += 1

            # Simple success-based sigma adaptation.
            # If improved, slightly increase sigma to keep exploration; else decrease.
            if improved:
                sigma *= 1.05
            else:
                sigma *= 0.85

            # Bound sigma.
            sigma = float(np.clip(sigma, 1e-12, 2.0 * float(np.mean(ranges))))

            # Restart if stuck.
            if stagnation >= stagnation_limit and restart_cooldown == 0 and evals < budget:
                # Restart population around a random point (global exploration).
                base = rng.uniform(low, high, size=d)
                # Larger sigma for restart.
                sigma = 0.5 * float(np.mean(ranges))
                X = np.clip(base + sigma * rng.standard_normal(size=(pop, d)), low, high)
                y = np.array([eval_x(X[i]) for i in range(pop) if evals < budget], dtype=float)
                # If we hit budget mid-restart, shrink to evaluated size.
                if y.shape[0] < pop:
                    pop_eff = y.shape[0]
                    X = X[:pop_eff]
                    pop = pop_eff
                if y.shape[0] == 0:
                    break
                cur_best_idx = int(np.argmin(y))
                if float(y[cur_best_idx]) < best_y:
                    best_y = float(y[cur_best_idx])
                    best_x = X[cur_best_idx].copy()
                stagnation = 0
                restart_cooldown = max(3, d // 2)

            if restart_cooldown > 0:
                restart_cooldown -= 1

            # Local refinement step near the best (only if budget remains).
            # Use a very small candidate batch for exploitation.
            if evals < budget:
                rem = budget - evals
                if rem > 0:
                    refine_lam = 1 if rem < 4 else min(2 + d // 6, 4)
                    refine_lam = min(refine_lam, rem)
                    if refine_lam > 0:
                        # Small neighborhood around best.
                        local_sigma = sigma * 0.2
                        local_sigma = max(local_sigma, 1e-12)
                        local = np.clip(
                            best_x[None, :] + local_sigma * rng.standard_normal(size=(refine_lam, d)),
                            low,
                            high,
                        )
                        for i in range(refine_lam):
                            if evals >= budget:
                                break
                            yi = eval_x(local[i])
                            if yi < best_y:
                                best_y = yi
                                best_x = local[i].copy()

        return best_x, best_y

    @staticmethod
    def _read_bounds(func):
        # Try func.lower/upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
            return lb, ub
        if hasattr(func, "bounds"):
            b = func.bounds
            # Typical objects: bounds.lb, bounds.ub
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return b.lb, b.ub
            if hasattr(b, "lower") and hasattr(b, "upper"):
                return b.lower, b.upper
        raise AttributeError(
            "Objective function must expose bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub"
        )
