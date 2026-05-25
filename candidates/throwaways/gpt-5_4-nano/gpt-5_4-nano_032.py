# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact derivative-free black-box minimizer
# using a randomized coordinate-direction (CMA-like) search with adaptive step
# sizes, elitist selection, and periodic local refinement. It works for any
# dimension and respects a hard evaluation budget by tracking every objective
# call.
# Search state: The algorithm maintains a current best point x_best, its value
# y_best, and an adaptive per-dimension step size vector sigma. It also keeps
# a small "population" of candidate points around the current best.
# Candidate generation: Each iteration samples several candidates by adding
# isotropic Gaussian noise scaled by sigma, plus a small coordinate-wise
# perturbation set. Samples are clipped to the provided bounds.
# Selection and replacement: After evaluating all candidates, the best candidate
# replaces x_best and updates y_best. The population is always centered around
# the current x_best (elitist).
# Adaptation: sigma is increased modestly after non-improving batches and
# decreased after improvements, using a success-based rule. Additionally,
# a simple axis-aligned refinement attempts greedy improvements along each
# coordinate direction with a shrinking step.
# Exploration mechanisms: Gaussian sampling provides global exploration; the
# coordinate perturbations encourage movement along axes and help escape
# plateaus.
# Exploitation mechanisms: When improvement is observed, sigma shrinks and a
# greedy coordinate refinement runs to polish the current best.
# Boundary handling: Candidate points are clipped to the feasible bounds at
# generation time. If bounds are degenerate in a dimension, sigma for that
# dimension is effectively forced to zero via clipping behavior.
# Budget strategy: The algorithm computes the number of evaluations per phase
# (batch size) from the total budget and ensures it never exceeds the remaining
# evaluations by truncating the final batch. A final refinement phase uses
# whatever evaluations remain.
# Closest known influences: The design is inspired by evolution strategies
# (fitness-based recombination) and adaptive step-size rules, combined with
# coordinate search refinement. The implementation is intentionally lightweight
# and standard-library-only besides numpy.
# Novelty or unusual aspects: The algorithm uses both batch Gaussian search and
# a budget-aware greedy coordinate refinement, with a success-rate-driven sigma
# update to make it robust on diverse bound-constrained functions.
# Failure modes: If the budget is extremely small, refinement may not run and
# the method may rely mostly on one batch of random samples. On highly
# ill-conditioned problems, clipping at bounds can reduce effective search
# directions; sigma adaptation mitigates this but cannot eliminate it entirely.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError(
                "Objective must expose bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )

        if lb.shape == ():  # scalar bounds
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        lb = lb.reshape(dim)
        ub = ub.reshape(dim)

        # Degenerate bounds are allowed; handle by defining range safely.
        span = ub - lb
        span = np.where(np.isfinite(span), span, 0.0)
        span = np.maximum(span, 0.0)

        # If budget is too small, still evaluate at least once.
        max_evals = max(1, budget)

        # ---- Evaluation accounting ----
        evals = 0

        def eval_at(x):
            nonlocal evals
            if evals >= max_evals:
                # Hard stop (should not happen if we budget-aware manage batches)
                return np.inf
            y = func(x)
            evals += 1
            return float(y)

        # ---- Initialization ----
        # Random starting point uniformly in bounds.
        # (Harness sets numpy seed before each run for reproducibility.)
        x0 = lb + np.random.rand(dim) * span
        x_best = np.clip(x0, lb, ub)
        y_best = eval_at(x_best)

        # Adaptive step size per dimension.
        # Start as a fraction of the span; if span is 0 in a dimension, sigma becomes 0.
        base = 0.2
        sigma = base * span
        # Avoid all-zero sigma (if bounds collapse everywhere, any x is the same).
        if not np.any(sigma > 0):
            sigma = np.zeros(dim, dtype=float)

        # If dimension is zero or budget exhausted, return best.
        if dim <= 0 or evals >= max_evals:
            return x_best, y_best

        # Helper: generate bounded samples around x_best.
        def sample_candidates(n, include_axis_moves=True):
            # Use Gaussian exploration around current best.
            # Candidate: x_best + sigma * N(0,1), clipped to bounds.
            candidates = []

            # If sigma is all zeros, sampling gives identical points; still works.
            # We'll fall back to axis moves which may also be no-ops.
            if n <= 0:
                return np.empty((0, dim), dtype=float)

            # First, Gaussian samples.
            n_gauss = n
            Z = np.random.randn(n_gauss, dim)  # n x dim
            X = x_best[None, :] + Z * sigma[None, :]

            X = np.clip(X, lb, ub)
            candidates.append(X)

            if include_axis_moves and n >= 2 and np.any(span > 0):
                # Add small axis moves for exploitation-ish exploration.
                # Choose a few coordinates randomly.
                k = min(dim, n)  # at most dim axis moves
                axes = np.random.choice(dim, size=k, replace=False) if dim >= k else np.arange(k)
                # Alternate signs, and scale by a fraction of span.
                signs = np.where(np.random.rand(k) < 0.5, -1.0, 1.0)
                step = (0.1 + 0.1 * np.random.rand(k)) * np.where(span[axes] > 0, span[axes], 0.0)
                X2 = np.tile(x_best, (k, 1))
                X2[np.arange(k), axes] = np.clip(
                    X2[np.arange(k), axes] + signs * step,
                    lb[axes],
                    ub[axes],
                )
                candidates.append(X2)

            # Stack and ensure we have exactly n candidates.
            all_cand = np.vstack(candidates) if len(candidates) > 1 else candidates[0]
            if all_cand.shape[0] > n:
                all_cand = all_cand[:n]
            return all_cand

        # ---- Search loop (budget-aware) ----
        # Choose a batch size that leaves room for refinement.
        # We adapt batch size for small budgets.
        # Minimum 4 evaluations per batch when possible.
        batch = min(16, max(4, max_evals // 10))
        # Keep at most a few batches; remaining budget used for refinement.
        # (We also ensure we don't exceed evals.)
        max_batches = 1
        if max_evals >= 40:
            max_batches = 6
        elif max_evals >= 20:
            max_batches = 4
        elif max_evals >= 10:
            max_batches = 3

        # We'll update until reaching budget.
        # Each batch uses up to (batch-1) new evaluations because x_best already evaluated.
        while evals < max_evals:
            # Stop if no evaluations left.
            remaining = max_evals - evals
            if remaining <= 0:
                break

            # Determine how many candidates we can evaluate this batch.
            # Each candidate costs 1 eval.
            # Ensure at least 1 candidate if possible.
            n_cand = min(batch, remaining)
            if n_cand <= 0:
                break

            # We already have x_best, so we generate n_cand new candidates.
            X = sample_candidates(n_cand, include_axis_moves=True)

            # Evaluate and select best.
            # If objective is noisy, this still works as it always keeps the best found.
            y_vals = np.empty(n_cand, dtype=float)
            for i in range(n_cand):
                y_vals[i] = eval_at(X[i])

            idx = int(np.argmin(y_vals))
            y_new = float(y_vals[idx])
            x_new = X[idx]

            improved = y_new < y_best

            if improved:
                x_best = x_new
                y_best = y_new
                # Success: shrink step size for exploitation but keep some exploration.
                # Reduce sigma more aggressively when improvement is good.
                # A simple scaling based on relative improvement.
                rel = (y_prev - y_best) / (abs(y_prev) + 1e-12) if 'y_prev' in locals() else 0.0
                # rel can be negative on non-improvement; only use when improved
                # but clamp for stability.
                rel = float(np.clip(rel, 0.0, 1.0))
                factor = 0.85 - 0.15 * rel  # between 0.7 and 0.85
                sigma *= factor
                sigma = np.maximum(sigma, 0.0)
            else:
                # Non-success: modestly increase sigma to encourage exploration.
                # Increase only if sigma has any mass; else keep it.
                if np.any(sigma > 0):
                    sigma *= 1.10
                sigma = np.maximum(sigma, 0.0)

            # Store y_prev for next iteration's relative improvement computation.
            y_prev = y_new

            # Optional refinement (coordinate greedy search) with small fraction of remaining budget.
            # Run it only occasionally to save budget.
            remaining = max_evals - evals
            if remaining <= 0:
                break
            # Trigger refinement when we improved or early in the run.
            should_refine = improved or (evals < max_evals * 0.4 and np.random.rand() < 0.5)

            if should_refine and remaining >= 2:
                # Use up to a small portion of remaining budget.
                refine_budget = min(10, remaining)
                # We'll attempt up to min(dim, refine_budget) coordinates.
                k = min(dim, refine_budget)
                # Greedy refinement tries to move along each coordinate in both directions
                # using a shrinking step.
                # Cost: k evaluations (one per coordinate) after selecting a direction.
                # If k*2 would exceed budget, we restrict to one direction per coordinate.
                coords = np.random.choice(dim, size=k, replace=False) if dim >= k else np.arange(k)

                # Current step scale for refinement.
                # Use a fraction of sigma/span, but avoid zero.
                step_scale = 0.25
                for j in coords:
                    if evals >= max_evals:
                        break
                    # If span[j] is zero, coordinate is fixed.
                    if span[j] <= 0 or sigma[j] <= 0:
                        continue

                    cur = x_best[j]
                    # Try a signed step that points towards whichever side currently seems more useful.
                    # We'll evaluate two directions only if budget allows; otherwise do one.
                    d = max(step_scale * sigma[j], 1e-12)
                    # Two directions (budget-aware)
                    if evals + 2 <= max_evals:
                        x1 = x_best.copy()
                        x2 = x_best.copy()
                        x1[j] = np.clip(cur + d, lb[j], ub[j])
                        x2[j] = np.clip(cur - d, lb[j], ub[j])
                        y1 = eval_at(x1)
                        y2 = eval_at(x2)
                        if y1 < y_best or y2 < y_best:
                            if y1 <= y2:
                                x_best = x1
                                y_best = y1
                            else:
                                x_best = x2
                                y_best = y2
                            # Successful refinement: shrink sigma a bit.
                            sigma *= 0.9
                    else:
                        # Budget tight: single direction.
                        sign = -1.0 if np.random.rand() < 0.5 else 1.0
                        x1 = x_best.copy()
                        x1[j] = np.clip(cur + sign * d, lb[j], ub[j])
                        y1 = eval_at(x1)
                        if y1 < y_best:
                            x_best = x1
                            y_best = y1
                            sigma *= 0.9

                    # Shrink d to make refinement progressively local.
                    sigma[j] *= 0.95

            # If we have evaluated most of the budget, exit.
            if evals >= max_evals:
                break

            # If batch loop count gets too large, break.
            # (This keeps runtime controlled.)
            max_batches -= 1
            if max_batches <= 0 and evals < max_evals:
                # Use remaining budget with one last exploratory batch and stop.
                remaining = max_evals - evals
                if remaining > 0:
                    n_cand = min(batch, remaining)
                    X = sample_candidates(n_cand, include_axis_moves=True)
                    y_vals = np.empty(n_cand, dtype=float)
                    for i in range(n_cand):
                        y_vals[i] = eval_at(X[i])
                    idx = int(np.argmin(y_vals))
                    if float(y_vals[idx]) < y_best:
                        x_best = X[idx]
                        y_best = float(y_vals[idx])
                break

        return x_best, y_best
