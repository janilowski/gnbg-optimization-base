# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm that uses a
# trust-region style local search plus global exploration. It maintains a
# small population of candidate solutions, iteratively improves the best
# solution, and occasionally expands exploration when progress stalls.
#
# Search state: Tracks the current best point x_best and its value y_best.
# Also keeps a "center" point for local search (the current best) and a step
# size / trust radius sigma that controls the scale of candidate perturbations.
# A small population is sampled around the center each iteration.
#
# Candidate generation: Each iteration proposes multiple candidates by sampling
# Gaussian perturbations around the local center with standard deviation
# sigma. With some probability, it also proposes points via wider
# "global" exploration (larger sigma) and uses occasional random resets to
# help escape stagnation.
#
# Selection and replacement: Among all evaluated candidates in an iteration
# (plus optionally the center), the algorithm keeps the best one as the new
# x_best/x_center. The population is not explicitly stored; instead, the
# algorithm uses the best found so far to guide subsequent sampling.
#
# Adaptation: If the best value improves, sigma is shrunk slightly to focus
# the search; if there's little or no improvement for several iterations, sigma
# is expanded and/or the search center is re-randomized to recover exploration.
#
# Exploration mechanisms: Larger-step sampling and periodic random points
# across the full bounds, controlled by stagnation detection.
#
# Exploitation mechanisms: Gaussian local sampling around the current best with
# adaptive sigma (shrinking on improvement) to refine the minimum.
#
# Boundary handling: All candidates are clipped to the provided bounds. If bounds
# are degenerate (zero width), the corresponding coordinate is fixed.
#
# Budget strategy: The algorithm computes the number of iterations and
# candidates per iteration so it never evaluates beyond the provided budget.
# It uses an "evaluate remaining budget" guard before each objective call.
#
# Closest known influences: A simplified evolution-strategy / CMA-lite style
# approach (population sampling + adaptive step size) blended with a
# trust-region heuristic and stagnation-based exploration.
#
# Novelty or unusual aspects: Keeps the implementation intentionally small and
# robust without covariance adaptation; uses clipped Gaussian steps plus
# periodic global probes guided by recent improvement.
#
# Failure modes: If the objective is extremely noisy or adversarial, sigma
# adaptation may oscillate and the algorithm may converge slowly. If bounds
# are very tight or degenerate, progress may be limited to the feasible region
# and the best found could remain near the initial probes.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from func ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = func.bounds.lb
            ub = func.bounds.ub
        else:
            raise AttributeError("func must provide bounds via lower/upper or bounds.lb/bounds.ub")

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != dim or ub.size != dim:
            # Allow scalar bounds for convenience
            if lb.size == 1:
                lb = np.full(dim, float(lb.item()), dtype=float)
            if ub.size == 1:
                ub = np.full(dim, float(ub.item()), dtype=float)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimensionality mismatch with dim")

        # Ensure correct ordering; if swapped, fix.
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo

        # Handle degenerate bounds (zero span): candidate coords fixed by clipping anyway.
        # Precompute an initial sigma scale.
        default_sigma = 0.3 * (np.mean(span) if np.any(span != 0) else 1.0)
        default_sigma = float(default_sigma) if np.isfinite(default_sigma) and default_sigma > 0 else 1.0

        # ---- Budget bookkeeping ----
        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                # Must never exceed budget; return current best sentinel
                return y_best
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            y = float(func(x))
            evals += 1
            return y

        # ---- Initial sampling ----
        # Always evaluate at least one point if budget allows.
        rng = np.random

        # Choose a small initial set size.
        # Heuristic: more initial points for higher dimension, but bounded by budget.
        init_k = min(budget, max(1, 2 + dim // 3))
        # Start from a mix of random points and a mid-point.
        x_mid = lo + 0.5 * span

        # If bounds are degenerate, mid == lo == hi.
        candidates = [x_mid]
        while len(candidates) < init_k:
            u = rng.rand(dim)
            x = lo + u * span
            candidates.append(x)

        x_best = None
        y_best = None
        for x0 in candidates:
            if evals >= budget:
                break
            y0 = evaluate(x0)
            if y_best is None or y0 < y_best:
                x_best = np.clip(x0, lo, hi).copy()
                y_best = y0

        if x_best is None:
            # In case budget == 0, but should not happen in typical harnesses.
            x_best = np.clip(x_mid, lo, hi).copy()
            y_best = float(func(x_best)) if budget > 0 else float("inf")

        # ---- Main loop design ----
        # We will run until we are close to budget.
        # Each iteration evaluates a small "batch" of candidates.
        # Ensure we do not exceed the budget.
        # Batch size: 4..(2*dim capped), but limited by remaining budget.
        base_batch = 4 + dim // 4
        batch = int(max(3, min(base_batch, max(3, budget))))  # at least 3 when possible

        # Iteration count derived from batch size; recomputed with remaining.
        sigma = default_sigma
        success_streak = 0
        fail_streak = 0
        # Stagnation threshold adapts to dim and budget.
        stagnation_limit = int(max(3, min(12, 2 + dim // 5)))
        # Global exploration probability.
        p_global = 0.25

        # For scaling sigma with bounds:
        # keep sigma relative to span when possible
        span_scale = float(np.mean(span)) if np.any(span != 0) else 1.0
        if not np.isfinite(span_scale) or span_scale <= 0:
            span_scale = 1.0

        # ---- Optimization loop ----
        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break

            # Adjust batch to remaining budget to avoid overruns.
            b = min(batch, remaining)

            # Local center: current best.
            x_center = x_best

            # Track best of this iteration.
            y_iter_best = y_best
            x_iter_best = x_best

            # Adapt sigma bounds to the feasible region scale.
            # Upper cap helps avoid wasting evaluations far away when bounds are tight.
            sigma_min = 1e-12 * (span_scale if span_scale > 0 else 1.0)
            sigma_max = 1.0 * (span_scale if span_scale > 0 else 1.0)
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 0.3 * span_scale
            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # Evaluate a batch of candidates.
            # - Most are local Gaussian around x_center with scale sigma.
            # - Some are global probes with larger sigma.
            for i in range(b):
                if evals >= budget:
                    break

                # Choose exploration mode.
                # Also increase global probability a bit when failing.
                pf = p_global + 0.15 * (fail_streak > 0)
                use_global = (rng.rand() < pf)

                if use_global:
                    # Global probe: mix of random point and broad perturbation.
                    if rng.rand() < 0.5:
                        u = rng.rand(dim)
                        x = lo + u * span
                    else:
                        # Broad perturbation around center
                        scale = 2.0 + 2.0 * rng.rand()
                        x = x_center + (sigma * scale) * rng.randn(dim)
                else:
                    # Exploitation probe: local Gaussian perturbation
                    # Use per-coordinate anisotropy based on bounds span to be robust.
                    # Coordinates with zero span are effectively fixed by clipping.
                    coord_scale = span
                    # Normalize coord_scale to avoid huge steps when span varies a lot.
                    finite = np.isfinite(coord_scale)
                    if np.any(finite) and np.any(coord_scale > 0):
                        # scale each coordinate relative to mean positive span
                        pos = coord_scale[finite] > 0
                        mean_pos = float(np.mean(coord_scale[finite][pos])) if np.any(pos) else span_scale
                        if mean_pos > 0 and np.isfinite(mean_pos):
                            coord_scale = np.where(coord_scale > 0, coord_scale / mean_pos, 0.0)
                        else:
                            coord_scale = np.where(coord_scale > 0, 1.0, 0.0)
                    else:
                        coord_scale = np.ones(dim, dtype=float)

                    x = x_center + (sigma * coord_scale) * rng.randn(dim)

                # Clip to bounds, evaluate
                y = evaluate(x)
                if y < y_iter_best:
                    y_iter_best = y
                    x_iter_best = np.clip(x, lo, hi).copy()

            # Selection: update global best if improved.
            if y_iter_best < y_best - 0.0:
                improvement = (y_best - y_iter_best)
                x_best, y_best = x_iter_best, y_iter_best
                success_streak += 1
                fail_streak = 0

                # Exploitation: shrink sigma on success
                # More aggressive shrinking with consecutive successes.
                shrink = 0.82 ** min(3, success_streak)
                sigma *= shrink

            else:
                fail_streak += 1
                success_streak = 0

                # Exploration / recovery: expand sigma on failure,
                # but gradually to avoid wild oscillations.
                expand = 1.15 ** min(3, fail_streak)
                sigma *= expand

                # If stagnating for long, perform a mild random reset around bounds.
                if fail_streak >= stagnation_limit:
                    fail_streak = 0
                    # Randomly move center closer to a likely region:
                    # either a random point or a midpoint perturbed by sigma.
                    if rng.rand() < 0.6:
                        u = rng.rand(dim)
                        x_reset = lo + u * span
                    else:
                        x_reset = x_best + sigma * rng.randn(dim)
                    x_reset = np.clip(x_reset, lo, hi)
                    # Use one evaluation to refresh best if budget permits.
                    if evals < budget:
                        y_reset = evaluate(x_reset)
                        if y_reset < y_best:
                            x_best, y_best = x_reset, y_reset
                            # reduce sigma after improvement
                            sigma *= 0.7
                    # Expand sigma to encourage escape
                    sigma = min(sigma_max, sigma * 1.5)

        return x_best, y_best
