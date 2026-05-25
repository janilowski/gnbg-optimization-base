# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# using a mixture of global sampling and local coordinate-wise improvement.
# It maintains a population of candidate solutions, repeatedly samples around
# the best-so-far point, and occasionally reinitializes to escape stagnation.
#
# Search state: The algorithm tracks best_x and best_y, and maintains a small
# population of current candidates with their function values. It also tracks
# remaining evaluations to ensure the budget is never exceeded, plus a simple
# stagnation counter to decide when to restart.
#
# Candidate generation: At each iteration, it samples new points by:
# (1) drawing a subset from the current population and applying small random
#     coordinate perturbations, and
# (2) sampling globally from the full bounds with small probability
#     (to preserve exploration).
#
# Selection and replacement: After evaluating new candidates, it merges them
# with the current population, keeps the best k by objective value, and updates
# best_x/best_y whenever improvements are found.
#
# Adaptation: The local perturbation step size adapts based on how much the
# best value improved recently: if progress is good, the step size is reduced
# (finer exploitation); if stagnation occurs, the step size increases and a
# restart may happen.
#
# Exploration mechanisms: Global random sampling from bounds and occasional
# restarts (replacing most of the population) provide exploration.
#
# Exploitation mechanisms: Local coordinate perturbations around promising
# points using an adaptive step size provide exploitation.
#
# Boundary handling: Every generated point is clipped to the provided bounds.
# If bounds are infinite/undefined, the code falls back to a safe default box
# derived from dim (but still respects function-provided finite bounds when
# available).
#
# Budget strategy: The total number of objective evaluations is capped exactly
# at the provided budget. The algorithm computes an evaluation budget for each
# phase and carefully stops early if the budget is exhausted.
#
# Closest known influences: The behavior resembles an adaptive evolutionary/
# evolution-strategy style (population + mutation + elitist selection) with
# coordinate-wise local search and restarts.
#
# Novelty or unusual aspects: The implementation uses simple coordinate-wise
# moves and adapts step size using only relative improvement, aiming for
# compactness and robustness across dimensions.
#
# Failure modes: If the objective is extremely noisy or the bounds are very
# narrow, progress may stall; restarts and global sampling mitigate this.
# If bounds are missing or infinite, the fallback box may be suboptimal.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return a trivial point.
            x0 = np.zeros(dim, dtype=float)
            return x0, float("inf")

        # ----- Read bounds -----
        lb = None
        ub = None

        # Preferred forms: func.lower/func.upper or func.bounds.lb/func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = b.lb
                ub = b.ub

        # Convert to arrays if possible, otherwise set a conservative fallback.
        def _to_array(v):
            if v is None:
                return None
            a = np.asarray(v, dtype=float)
            if a.ndim == 0:
                a = np.full(dim, float(a), dtype=float)
            return a

        lb = _to_array(lb)
        ub = _to_array(ub)

        finite_lb = np.isfinite(lb).all() if lb is not None else False
        finite_ub = np.isfinite(ub).all() if ub is not None else False

        if lb is None or ub is None or not (finite_lb and finite_ub):
            # Fallback: default box centered at 0 with a size depending on dim.
            # Keep it finite to enable clipping.
            # If one side exists and is finite, respect it; otherwise use fallback.
            default_scale = 1.0 + 0.1 * dim
            if lb is None or not np.isfinite(lb).all():
                lb = -default_scale * np.ones(dim, dtype=float)
            else:
                lb = lb.astype(float, copy=False)
            if ub is None or not np.isfinite(ub).all():
                ub = default_scale * np.ones(dim, dtype=float)
            else:
                ub = ub.astype(float, copy=False)

        # Ensure lb <= ub and shape correctness
        lb = np.asarray(lb, dtype=float).reshape(dim)
        ub = np.asarray(ub, dtype=float).reshape(dim)
        # In case the harness provides reversed bounds, swap
        swap_mask = lb > ub
        if np.any(swap_mask):
            lb2 = lb.copy()
            ub2 = ub.copy()
            lb2[swap_mask], ub2[swap_mask] = ub2[swap_mask], lb2[swap_mask]
            lb, ub = lb2, ub2

        width = ub - lb
        # Avoid zero widths; use small epsilon.
        width = np.where(width > 0, width, 1e-12)
        half_width = 0.5 * width

        # ----- Evaluation wrapper (budget-capped) -----
        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                return None
            x = np.asarray(x, dtype=float)
            # func is expected to accept 1D array and return scalar.
            y = func(x)
            evals += 1
            return float(y)

        # ----- Initialize population -----
        # Choose population size adaptively but within budget.
        # k is the number of elite individuals kept.
        k = max(2, min(10, dim + 1))
        pop_size = max(k, min(20, 2 * k, budget))  # total initial + elites around

        # Create random initial candidates uniformly in bounds.
        # If budget is small, we keep pop_size accordingly.
        pop_size = min(pop_size, budget)

        # Generate and evaluate initial population
        X = np.empty((pop_size, dim), dtype=float)
        ys = np.empty(pop_size, dtype=float)

        for i in range(pop_size):
            r = np.random.rand(dim)
            x = lb + r * (ub - lb)
            y = evaluate(x)
            if y is None:
                # Budget exceeded unexpectedly; shrink arrays.
                X = X[:i]
                ys = ys[:i]
                pop_size = i
                break
            X[i] = x
            ys[i] = y

        if pop_size == 0:
            x0 = lb.copy()
            y0 = evaluate(x0)
            return (x0 if y0 is not None else x0), (y0 if y0 is not None else float("inf"))

        # Sort by fitness (minimization)
        idx = np.argsort(ys)
        X = X[idx]
        ys = ys[idx]

        best_x = X[0].copy()
        best_y = float(ys[0])

        # Elite size for replacement
        k = min(k, pop_size)

        # Adaptive step size: start as a fraction of range.
        # Use per-dimension scaling.
        # Start relatively wide to explore, then shrink with progress.
        step = 0.25 * half_width
        # Ensure non-zero step
        step = np.where(step > 0, step, 1e-6)

        # Stagnation controls
        stagnation = 0
        best_improve_threshold = 1e-12
        max_stagnation = 8

        # Remaining evaluations
        # We'll run "rounds", each consuming a fixed number of evaluations.
        # Keep rounds count modest for compactness.
        remaining = budget - evals
        if remaining <= 0:
            return best_x, best_y

        # Determine number of new candidates per round
        # Aim to use most remaining evaluations efficiently.
        per_round = min(4 * k, 50, remaining)  # candidate count per round
        # Reduce if budget is tiny
        per_round = max(1, per_round)

        # A helper for generating a single offspring
        def make_offspring(parent, global_prob):
            # global_prob decides whether we sample globally or locally
            if np.random.rand() < global_prob:
                # Global exploration: uniform sampling in bounds
                r = np.random.rand(dim)
                x = lb + r * (ub - lb)
            else:
                # Local exploitation: coordinate-wise random perturbation
                # Choose a few coordinates to perturb.
                x = parent.copy()
                # Number of coordinates to change (1..dim but capped)
                # Larger dim => perturb more coordinates.
                m = 1 + int(np.random.rand() * min(dim, 6))
                coords = np.random.choice(dim, size=m, replace=False)
                # Direction with Gaussian noise scaled by step
                # Use symmetric perturbations
                noise = np.random.randn(m)
                # Scale by step and relative coordinate width
                x[coords] = x[coords] + noise * step[coords]

                # Optional: small additional isotropic jitter
                if np.random.rand() < 0.25:
                    x = x + np.random.randn(dim) * (0.05 * step)

            # Boundary handling: clip to [lb, ub]
            x = np.minimum(ub, np.maximum(lb, x))
            return x

        # Main loop
        while evals < budget:
            remaining = budget - evals
            # Stop if no evaluations left
            if remaining <= 0:
                break

            # Use a variable global exploration probability
            # Higher when stagnating.
            global_prob = 0.05 + 0.15 * (stagnation > 0) + 0.1 * (stagnation > 3)
            global_prob = min(0.35, global_prob)

            # Decide how many candidates to evaluate this round
            n_new = min(per_round, remaining)
            X_new = np.empty((n_new, dim), dtype=float)
            y_new = np.empty(n_new, dtype=float)

            # Choose parents from elites with fitness-proportional bias to best
            # (lower y => higher chance).
            elite_X = X[:k]
            elite_y = ys[:k]
            # Convert to weights: inverse of (y - min + epsilon)
            y_shift = elite_y - elite_y.min()
            weights = 1.0 / (y_shift + 1e-12)
            weights = weights / weights.sum()

            # Adaptive parent step: sometimes perturb best more aggressively.
            for i in range(n_new):
                # Sample a parent index
                pi = int(np.random.choice(k, p=weights))
                parent = elite_X[pi]
                # Sometimes use best directly (strong exploitation)
                if np.random.rand() < 0.2:
                    parent = best_x
                child = make_offspring(parent, global_prob)
                y = evaluate(child)
                if y is None:
                    X_new = X_new[:i]
                    y_new = y_new[:i]
                    n_new = i
                    break
                X_new[i] = child
                y_new[i] = y

            if n_new <= 0:
                break

            # Combine and select elites
            X_comb = np.vstack((X, X_new))
            y_comb = np.concatenate((ys, y_new))
            order = np.argsort(y_comb)
            X = X_comb[order][:k]
            ys = y_comb[order][:k]

            # Update global best and adaptation
            new_best = float(ys.min())
            if new_best + best_improve_threshold < best_y:
                # Improvement found
                prev_best_y = best_y
                best_y = float(new_best)
                best_x = X[np.argmin(ys)].copy()

                # If improvement is "substantial", reduce step size to exploit
                # Otherwise keep moderate.
                rel = abs(prev_best_y - best_y) / (abs(prev_best_y) + 1e-9)
                if rel > 0.01:
                    step = step * 0.85
                else:
                    step = step * 0.92
                stagnation = 0
            else:
                stagnation += 1
                # If no improvement, increase step slightly to diversify
                step = step * (1.10 + 0.05 * min(3, stagnation))
                # Cap step to range
                step = np.minimum(step, half_width)

            # Restart mechanism if stagnation persists
            if stagnation >= max_stagnation and evals < budget:
                # Reinitialize most population with global samples.
                # Keep the current best to preserve exploitation.
                keep = 1
                # Generate new candidates for remaining elites
                m = k - keep
                if m > 0:
                    for i in range(m):
                        # Use clipped uniform sampling
                        r = np.random.rand(dim)
                        x = lb + r * (ub - lb)
                        y = evaluate(x)
                        if y is None:
                            break
                        # Replace the worst in current elites
                        worst_idx = k - 1
                        X[worst_idx] = x
                        ys[worst_idx] = y
                    # Resort elites
                    order = np.argsort(ys)
                    X = X[order]
                    ys = ys[order]
                    # Update global best if needed
                    if float(ys[0]) + best_improve_threshold < best_y:
                        best_y = float(ys[0])
                        best_x = X[0].copy()
                # Make sure step is fairly large after restart
                step = np.maximum(step, 0.25 * half_width)
                stagnation = 0

        return best_x, best_y
