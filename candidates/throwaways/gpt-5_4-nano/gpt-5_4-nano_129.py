# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy combining
# random restarts, coordinate-wise local refinement, and simple step-size
# control. It is designed to be robust across dimensions while staying within
# a strict evaluation budget.
# Search state: Maintains a current best point (best_x, best_y), a step size
# sigma, and a counter of evaluations used. Uses a small set of candidate
# points sampled around the best solution.
# Candidate generation: At each iteration, generates candidates by adding
# Gaussian perturbations scaled by sigma, plus additional candidates formed
# by coordinate-wise perturbations. Also uses occasional random samples for
# global exploration.
# Selection and replacement: Evaluates candidates (without exceeding budget),
# then selects the best among them as the new best solution if it improves.
# Also performs a local "coordinate sweep" refinement around the current best
# using a shrinking step when improvements are found.
# Adaptation: If improvements occur, sigma is modestly increased; if not,
# sigma is reduced (multiplicative step-size control).
# Exploration mechanisms: Random restarts / pure random samples are triggered
# early and when progress stalls. Additionally, Gaussian sampling retains
# stochastic exploration.
# Exploitation mechanisms: Candidate distributions center around best_x, and
# coordinate-wise refinement attempts to reduce the objective along individual
# dimensions.
# Boundary handling: All samples are clipped to the provided box constraints.
# If bounds are degenerate for a dimension, sampling respects that fixed value.
# Budget strategy: The algorithm carefully tracks remaining evaluations and
# never exceeds the provided budget. It uses a fixed per-iteration candidate
# count derived from remaining budget and dimension.
# Closest known influences: Inspired by CMA-like random sampling and classical
# derivative-free patterns (evolutionary strategy / coordinate descent), but
# simplified for standard-library-only implementation.
# Novelty or unusual aspects: Combines a global-ish Gaussian sampler with a
# lightweight coordinate sweep that is invoked as a refinement step, using the
# same sigma schedule.
# Failure modes: If the objective is extremely noisy or highly irregular, the
# step-size adaptation and coordinate refinement may waste evaluations without
# converging. With very small budgets, it primarily relies on initial random
# sampling plus a minimal local search.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # ---- Bounds handling ----
        lb, ub = None, None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb, ub = func.lower, func.upper
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb, ub = b.lb, b.ub

        n = self.dim
        if lb is None or ub is None:
            lb = -np.ones(n, dtype=float)
            ub = np.ones(n, dtype=float)

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != n or ub.size != n:
            # Fall back gracefully if shapes mismatch
            lb = np.resize(lb, n).astype(float)
            ub = np.resize(ub, n).astype(float)

        # Ensure lb <= ub
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)

        # Handle degenerate bounds: enforce exact coordinate where lo==hi
        fixed_mask = hi == lo
        span = hi - lo
        # Avoid divide-by-zero; for fixed dims, span=0.
        safe_span = np.where(span > 0, span, 1.0)

        def clip(x):
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                x = np.minimum(np.maximum(x, lo), hi)
                if np.any(fixed_mask):
                    x[fixed_mask] = lo[fixed_mask]
                return x
            # not expected to happen; keep robust
            x = np.minimum(np.maximum(x, lo), hi)
            if np.any(fixed_mask):
                x[..., fixed_mask] = lo[fixed_mask]
            return x

        # ---- Objective evaluation with budget guard ----
        evals = 0
        best_x = None
        best_y = np.inf

        def evaluate(x):
            nonlocal evals, best_x, best_y
            if evals >= self.budget:
                # Should never happen if logic is correct
                return best_y
            y = float(func(np.asarray(x, dtype=float)))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.asarray(x, dtype=float).copy()
            return y

        # ---- Initialization ----
        rng = np.random  # harness sets np.random.seed externally
        remaining = self.budget

        # A reasonable initial sigma: fraction of box size (per-dimension),
        # but use a scalar sigma for simplicity.
        # For fixed dims, contribution is 0.
        box_size = float(np.max(safe_span))
        if not np.isfinite(box_size) or box_size <= 0:
            box_size = 1.0
        sigma = 0.25 * box_size

        # Initial sampling: spend a small portion (or up to budget) on random points.
        # Keep it compact: at least 1, at most n+1, and never exceed budget.
        init_count = min(self.budget, max(2, min(n + 1, self.budget)))
        # If budget is tiny, just evaluate once.
        if init_count > 0 and remaining > 0:
            for _ in range(init_count):
                # uniform sample in bounds
                u = rng.uniform(0.0, 1.0, size=n)
                x = lo + u * (hi - lo)
                x = clip(x)
                evaluate(x)

        # If no evaluations (budget==0), return default
        if evals == 0:
            x0 = clip((lo + hi) * 0.5)
            return x0, float(func(x0))

        # If budget exhausted, return best
        if evals >= self.budget:
            return best_x, best_y

        # ---- Main loop ----
        # Candidate counts are budget-aware: we generate batch candidates per round.
        # Local refinement is triggered when a better point is found.
        last_improve_iter = 0
        iter_idx = 0

        # Limits to prevent too much overhead in higher dimensions
        max_batch = min(self.budget - evals, 12 + n // 2)

        while evals < self.budget:
            iter_idx += 1
            remaining = self.budget - evals

            # Decide batch size: smaller near budget end
            batch = min(max_batch, max(2, min(remaining, 6 + n // 3)))

            # Exploration probability decreases over time
            # but still ensures some global sampling.
            progress = evals / max(1, self.budget)
            p_explore = 0.35 * (1.0 - progress) + 0.05

            candidates = []

            # Occasionally add a pure random point to escape stagnation.
            if rng.rand() < p_explore:
                u = rng.uniform(0.0, 1.0, size=n)
                candidates.append(clip(lo + u * (hi - lo)))

            # Fill rest with Gaussian perturbations around best_x
            # Gaussian noise scaled by sigma and (optionally) per-dimension span.
            # For fixed dims, span==0 so these perturbations stay at the fixed value,
            # and we still clip for safety.
            while len(candidates) < batch:
                z = rng.normal(0.0, 1.0, size=n)
                # Use per-dimension scaling to respect coordinate ranges
                step = sigma * (safe_span / box_size)
                x = best_x + z * step
                candidates.append(clip(x))

            # Evaluate candidates and find best in the batch
            y_before = best_y
            best_local_x = best_x
            best_local_y = best_y

            for x in candidates:
                if evals >= self.budget:
                    break
                y = evaluate(x)
                # evaluate already updates global best, but track local too
                if y < best_local_y:
                    best_local_y = y
                    best_local_x = np.asarray(x, dtype=float).copy()

            improved = best_local_y < y_before
            if improved:
                last_improve_iter = iter_idx
                # Moderate sigma expansion on improvement
                sigma *= 1.15
                # Coordinate-wise refinement: try +/- along selected coordinates
                # using a reduced step to avoid massive jumps.
                # Only do it when we have enough budget to be meaningful.
                refine_budget = self.budget - evals
                if refine_budget > 0:
                    # Choose a subset of coordinates (covers all for small dims)
                    k = min(n, max(2, int(np.ceil(np.sqrt(n)))))
                    # Random subset
                    coords = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)

                    # Coordinate step decreases with dimension and progress
                    coord_step = sigma * 0.2
                    # Evaluate at most 2*k points but cap by remaining budget
                    max_points = min(refine_budget, 2 * k)

                    # Order coordinates by current bound span (larger spans get attention)
                    # to adapt to heterogeneous scales.
                    span_strength = np.abs(safe_span[coords])
                    order = np.argsort(-span_strength)  # descending
                    coords = coords[order]

                    points_tested = 0
                    for j in coords:
                        if points_tested >= max_points or evals >= self.budget:
                            break
                        # Try + direction
                        x1 = best_x.copy()
                        delta = coord_step * (safe_span[j] / box_size)
                        x1[j] = np.clip(x1[j] + delta, lo[j], hi[j])
                        if fixed_mask[j]:
                            x1[j] = lo[j]
                        if points_tested < max_points and evals < self.budget:
                            evaluate(x1)
                            points_tested += 1
                            if evals >= self.budget:
                                break
                        # Try - direction
                        if points_tested >= max_points:
                            break
                        x2 = best_x.copy()
                        x2[j] = np.clip(x2[j] - delta, lo[j], hi[j])
                        if fixed_mask[j]:
                            x2[j] = lo[j]
                        if points_tested < max_points and evals < self.budget:
                            evaluate(x2)
                            points_tested += 1

                    # If refinement didn't help, shrink a bit to focus
                    if best_y >= y_before:
                        sigma *= 0.7
            else:
                # No improvement: reduce sigma to zoom in
                sigma *= 0.82

            # If stagnation, do a stronger exploration by resetting sigma
            if iter_idx - last_improve_iter >= max(3, n // 2):
                # Occasional random reseeding around center to regain diversity
                if evals < self.budget:
                    # Set sigma back to a larger scale within bounds
                    sigma = min(0.5 * box_size, sigma * 1.5 + 1e-12)
                    # One random sample near center (not fully uniform to keep some structure)
                    u = rng.uniform(0.0, 1.0, size=n)
                    center = (lo + hi) * 0.5
                    x = center + (u - 0.5) * (0.8 * (hi - lo))
                    x = clip(x)
                    evaluate(x)
                last_improve_iter = iter_idx

            # Hard safety: sigma cannot be negative or NaN
            if not np.isfinite(sigma) or sigma <= 1e-15:
                sigma = max(1e-12, 0.01 * box_size)

        return np.asarray(best_x, dtype=float), float(best_y)
