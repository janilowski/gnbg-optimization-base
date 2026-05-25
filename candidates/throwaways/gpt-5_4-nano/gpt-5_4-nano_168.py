# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm that works
# for arbitrary dimensions. It combines a small population of candidates with a
# Gaussian sampling around the current best solution, plus a fallback global-ish
# restart mechanism to recover when progress stalls.
# Search state: Maintains a population of points, their function values, the current
# best point/value, and an evaluation counter that never exceeds the provided budget.
# Candidate generation: Initially samples uniformly within bounds. After that, repeatedly
# generates new candidates by perturbing the current best with isotropic Gaussian noise.
# Selection and replacement: Uses elitist replacement: after evaluating a batch, it keeps
# the best individuals, updates the global best, and replaces the worst individuals with
# newly sampled candidates.
# Adaptation: The Gaussian step size adapts based on whether the best value improves.
# Exploration mechanisms: A restart triggers when no improvement has occurred for a while,
# sampling a fresh population around a random point and resetting step size.
# Exploitation mechanisms: Most iterations sample around the current best with shrinking
# step size to focus search.
# Boundary handling: Every candidate is clipped to the provided bounds before evaluation.
# Budget strategy: Counts every objective call and ensures total evaluations never exceed
# `budget`. The algorithm uses mini-batches sized to fit remaining budget.
# Closest known influences: Inspired by evolutionary strategies / CMA-like intuitions
# but simplified to an isotropic (diagonal) self-adaptive sampling around the elite.
# Novelty or unusual aspects: Uses a robust, dimension-agnostic isotropic step adaptation
# combined with a stall-based restart, designed to be safe under strict evaluation budgets.
# Failure modes: If the objective is extremely noisy or discontinuous, Gaussian local
# sampling may struggle; in that case restarts help but the budget may still limit performance.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lower, upper = self._read_bounds(func, self.dim)
        dim = self.dim

        # Handle degenerate dimensions / bounds safely
        span = upper - lower
        span = np.where(np.isfinite(span) & (span > 0), span, 0.0)
        # If span is zero in a dimension, sampling that coordinate should be fixed.
        # We'll allow step size to respect zeros.

        # Budget: ensure at least 1 eval if budget >= 1.
        max_evals = max(0, self.budget)
        if max_evals == 0:
            # No evaluation possible; return a point at lower bound (no objective calls).
            x0 = lower.copy()
            return x0, float("inf")

        # Helper for clipping
        def clip(x):
            return np.minimum(np.maximum(x, lower), upper)

        # Evaluate with strict budget enforcement
        evals = 0

        # Choose evaluation batch sizes that fit budget and keep overhead low.
        pop_size = int(np.clip(4 + 2 * dim, 8, 40))
        pop_size = min(pop_size, max_evals)

        # Initial population: uniform sampling within bounds
        U = np.random.rand(pop_size, dim)
        X = lower + U * span
        X = clip(X)

        # Evaluate initial population
        Y = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            if evals >= max_evals:
                Y = Y[:i]
                X = X[:i]
                break
            y = func(X[i])
            evals += 1
            Y[i] = float(y)

        # Track best
        best_idx = int(np.argmin(Y))
        best_x = X[best_idx].copy()
        best_y = float(Y[best_idx])

        # Set initial step size relative to search space
        # Start larger for big dims but keep stable across scales.
        # If span is mostly zero, step will be near zero.
        step = 0.5 * np.mean(span) if np.mean(span) > 0 else 0.0
        if step == 0.0:
            # If all spans are zero, every point is identical.
            return best_x, best_y

        # Stall-based restart
        no_improve = 0
        patience = max(10, int(5 + dim * 0.5))
        min_step = 1e-12 * (np.mean(span) + 1e-12)

        # Utility: evaluate a batch safely
        def eval_batch(cand):
            nonlocal evals
            m = cand.shape[0]
            vals = np.empty(m, dtype=float)
            for j in range(m):
                if evals >= max_evals:
                    return cand[:j], vals[:j]
                vals[j] = float(func(cand[j]))
                evals += 1
            return cand, vals

        # Main loop: until budget exhausted
        # We'll use elites and isotropic Gaussian sampling around best_x.
        while evals < max_evals:
            remaining = max_evals - evals
            # Number of new candidates to propose this iteration
            m = int(min(pop_size, remaining))
            # Use a mixture: most samples around best_x, some uniform for exploration
            # (kept small to conserve budget).
            explore_k = max(1, m // 10)  # ~10% exploration
            exploit_k = m - explore_k

            # Exploit: Gaussian around best
            # Isotropic gaussian; scaled by step and span to adapt to coordinate ranges.
            # Use span-normalized noise so that each coordinate sees similar relative movement.
            # If span_i==0, that coordinate won't move.
            if exploit_k > 0:
                eps = np.random.randn(exploit_k, dim)
                scale_vec = (span / (np.mean(span) + 1e-18))  # normalize spans
                cand_exploit = best_x + (step * scale_vec) * eps
            else:
                cand_exploit = np.empty((0, dim))

            # Explore: uniform within bounds (near-random restarts)
            if explore_k > 0:
                U = np.random.rand(explore_k, dim)
                cand_explore = lower + U * span
            else:
                cand_explore = np.empty((0, dim))

            cand = np.vstack([cand_exploit, cand_explore]) if m > 0 else np.empty((0, dim))
            if cand.shape[0] == 0:
                break

            cand = clip(cand)

            cand, cand_y = eval_batch(cand)
            if cand_y.size == 0:
                break

            # Combine population + new candidates
            # Use elitist strategy to keep best pop_size individuals
            X_all = np.vstack([X, cand]) if X.size else cand.copy()
            Y_all = np.hstack([Y, cand_y]) if Y.size else cand_y.copy()

            # Sort by objective (minimization)
            order = np.argsort(Y_all)
            X_all = X_all[order]
            Y_all = Y_all[order]

            # Keep top pop_size or all if fewer than pop_size
            keep = min(pop_size, X_all.shape[0])
            X = X_all[:keep].copy()
            Y = Y_all[:keep].copy()

            # Update global best
            cur_best_idx = int(np.argmin(Y))
            cur_best_y = float(Y[cur_best_idx])
            cur_best_x = X[cur_best_idx].copy()

            if cur_best_y < best_y - 1e-12 * (abs(best_y) + 1.0):
                best_y = cur_best_y
                best_x = cur_best_x
                no_improve = 0
                # Successful iteration: shrink a bit but not too fast
                step = max(min_step, step * (0.75 + 0.15 * np.random.rand()))
            else:
                no_improve += 1
                # Unsuccessful: increase or maintain exploration by slightly increasing step,
                # then clamp to reasonable size.
                step = max(min_step, step * (0.95 + 0.2 * np.random.rand()))
                # Prevent runaway: cap based on bounds span
                step = min(step, 0.5 * (np.mean(span) + 1e-18))

            # Restart if stalled: reinitialize population around random points
            if no_improve >= patience and evals < max_evals:
                # Fresh center: pick best-known with some probability, else random
                if np.random.rand() < 0.4:
                    center = best_x.copy()
                else:
                    U = np.random.rand(dim)
                    center = lower + U * span
                # Reset population by sampling around center with larger step
                step = max(step, 0.5 * (np.mean(span) + 1e-18))
                U = np.random.randn(pop_size, dim)
                X_new = center + (step * (span / (np.mean(span) + 1e-18))) * U
                X_new = clip(X_new)

                # Evaluate new population until budget ends
                X, Y = None, None
                cand, cand_y = eval_batch(X_new)
                if cand_y.size == 0:
                    break
                X = cand.copy()
                Y = cand_y.copy()
                cur_best_idx = int(np.argmin(Y))
                best_x = X[cur_best_idx].copy()
                best_y = float(Y[cur_best_idx])
                no_improve = 0
                # After restart, reduce step for exploitation
                step = max(min_step, step * 0.5)

        return best_x, best_y

    def _read_bounds(self, func, dim):
        # Try multiple conventions:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        lower = upper = None

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = getattr(func, "lower")
            upper = getattr(func, "upper")
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lower = func.bounds.lb
            upper = func.bounds.ub
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lower") and hasattr(func.bounds, "upper"):
            lower = func.bounds.lower
            upper = func.bounds.upper

        if lower is None or upper is None:
            raise AttributeError(
                "Objective function must provide bounds via "
                "`func.lower/func.upper` or `func.bounds.lb/func.bounds.ub` (or similar)."
            )

        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        if lower.size == 1 and dim > 1:
            lower = np.full(dim, float(lower[0]))
        if upper.size == 1 and dim > 1:
            upper = np.full(dim, float(upper[0]))
        if lower.size != dim or upper.size != dim:
            raise ValueError(f"Bounds dimension mismatch: expected dim={dim}, got lb={lower.size}, ub={upper.size}")
        # Ensure numerical validity
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("Bounds contain non-finite values.")
        # If any bounds are reversed, swap them for robustness.
        lo = np.minimum(lower, upper)
        hi = np.maximum(lower, upper)
        return lo, hi
