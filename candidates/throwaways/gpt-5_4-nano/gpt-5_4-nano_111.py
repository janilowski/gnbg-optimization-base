# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, robust black-box minimization
# algorithm for continuous domains using a derivative-free randomized
# search. It combines global sampling, local coordinate-wise refinement,
# and a simple adaptive step-size based on recent improvements.
#
# Search state: The algorithm maintains the current best solution x_best and
# its objective value y_best. It also tracks an evaluation budget remaining
# (never exceeded) and an adaptive mutation scale (step_size) that shrinks
# when progress stalls and grows slightly after improvements.
#
# Candidate generation: Each iteration proposes candidates by (1) uniform
# random sampling over the bounds for exploration and (2) local perturbations
# around x_best using Gaussian noise scaled by step_size. Additionally, when
# local improvement is promising, it performs a coordinate-wise one-dimensional
# scan along each axis (using a small number of steps) to refine x_best.
#
# Selection and replacement: All proposed points are evaluated and the best
# among them updates x_best/x_best_y. The algorithm also records whether any
# candidate improved the current best to drive adaptation.
#
# Adaptation: step_size starts at a fraction of the domain width and is adapted:
# it decreases when improvements are rare, and increases modestly after an
# improvement to escape local traps. A lower bound prevents it from shrinking to
# numerical irrelevance.
#
# Exploration mechanisms: Random global sampling and occasional larger
# perturbations (controlled by step_size) help explore widely.
#
# Exploitation mechanisms: Gaussian local search around the best point plus
# coordinate-wise refinement focuses on improving the current best.
#
# Boundary handling: Candidates are clipped to the [lb, ub] bounds. If bounds
# are degenerate (lb == ub), those coordinates remain fixed naturally.
#
# Budget strategy: The algorithm uses a fixed iteration loop computed from the
# provided evaluation budget. It carefully decrements the remaining evaluation
# count and stops once the budget is exhausted. Every objective call counts
# toward the budget; the code never evaluates beyond the budget.
#
# Closest known influences: The design is loosely inspired by common derivative-
# free strategies such as (1) evolution-strategy style Gaussian mutations and
# (2) coordinate-wise local refinement with adaptive step sizes.
#
# Novelty or unusual aspects: The implementation is intentionally compact and
# dimension-robust: it dynamically scales exploration batch sizes and the
# coordinate refinement effort to fit the evaluation budget.
#
# Failure modes: If the objective is extremely noisy or highly ill-conditioned,
# coordinate scans may waste evaluations; the adaptive step size and budget-aware
# tuning mitigate this. If bounds are very large, clipping may dominate and
# reduce effective movement.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        if dim <= 0:
            # Degenerate case: nothing to optimize.
            x_best = np.zeros(0, dtype=float)
            return x_best, float(func(x_best))

        # ---- Read bounds robustly from the function object ----
        lb = None
        ub = None
        # Common patterns: func.lower/func.upper OR func.bounds.lb/func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = getattr(func, "lower")
            ub = getattr(func, "upper")
        elif hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = getattr(b, "lb")
                ub = getattr(b, "ub")

        if lb is None or ub is None:
            raise AttributeError(
                "Objective function must provide bounds via func.lower/func.upper "
                "or func.bounds.lb/func.bounds.ub."
            )

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size == 1:
            lb = np.full(dim, lb.item(), dtype=float)
        if ub.size == 1:
            ub = np.full(dim, ub.item(), dtype=float)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds must be scalars or arrays of length dim.")

        # Ensure proper order (robustness)
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        lb, ub = lb2, ub2

        # Clip helper
        width = ub - lb
        # Handle degenerate widths gracefully
        width_safe = np.where(width > 0, width, 1.0)
        rng = np.random

        # ---- Budget accounting ----
        budget = max(0, int(self.budget))
        evals_used = 0

        def eval_obj(x):
            nonlocal evals_used
            if evals_used >= budget:
                # Should never happen; guard for robustness.
                return np.inf
            y = func(x)
            evals_used += 1
            return float(y)

        # If budget is 0, return something without evaluating (not meaningful).
        # But requirement says never exceed budget; doesn't say we must evaluate.
        if budget <= 0:
            x_best = np.clip(np.zeros(dim, dtype=float), lb, ub)
            return x_best, float("inf")

        # ---- Initialize: pick a reasonable starting point ----
        # Choose mid-point, plus one random point if budget allows.
        x_best = np.clip((lb + ub) / 2.0, lb, ub)
        y_best = eval_obj(x_best)

        if evals_used < budget:
            x0 = lb + rng.random(dim) * (ub - lb)
            y0 = eval_obj(x0)
            if y0 < y_best:
                x_best, y_best = x0, y0

        # Adaptive step size: a fraction of the domain width
        # Start with something that can move a meaningful fraction of the range.
        step_size = 0.25 * np.max(width_safe)
        min_step = 1e-12

        # Choose how many candidates per iteration; tuned to budget and dimension.
        # We evaluate in small batches to keep selection effective.
        # Coordinate refinement: up to 2 steps per axis, but budget-limited.
        def estimate_batch_size():
            # Rough heuristic: evaluate ~O(dim) points initially, then smaller later.
            # Ensure at least 1 and not too large.
            if dim <= 5:
                return 5
            if dim <= 20:
                return 3
            return 2

        batch_size = estimate_batch_size()
        # Number of outer iterations: ensure total evals won't exceed budget.
        # Each outer loop evaluates batch_size candidates + maybe coordinate scan.
        # We'll compute a conservative number.
        remaining = budget - evals_used
        if remaining <= 0:
            return x_best, y_best

        # Conservative max outer iterations based purely on batch evaluations.
        max_outer = max(1, remaining // max(1, batch_size))
        # We'll cap to a small multiple to allow refinement occasionally.
        max_outer = min(max_outer, 2000)

        # Coordinate refinement settings (budget-aware).
        coord_scan_steps = 2  # points per direction per axis (plus both directions)
        # Total coordinate evaluations: axes * coord_scan_steps * 2 (both directions) + optional baseline
        # We'll only perform coordinate scan when we have enough budget.
        base_axes_attempt = min(dim, 10)  # don't scan all axes when dim is large

        for outer in range(max_outer):
            if evals_used >= budget:
                break

            # ---- Exploration vs exploitation schedule ----
            # Increase exploration early, then focus locally.
            t = outer / max_outer
            explore_prob = 0.55 * (1.0 - t) + 0.15  # between ~0.15 and 0.7
            # Ensure we still exploit late.
            explore_prob = float(np.clip(explore_prob, 0.05, 0.85))

            improved = False
            best_local_x = x_best
            best_local_y = y_best

            # ---- Candidate batch ----
            # We generate a mix of uniform random points and Gaussian mutations
            # around the current best.
            for _ in range(batch_size):
                if evals_used >= budget:
                    break

                if rng.random() < explore_prob:
                    # Global exploration: uniform sample in bounds
                    x = lb + rng.random(dim) * (ub - lb)
                else:
                    # Local exploitation: Gaussian around best, scaled per-dimension by width
                    # and capped to avoid excessive excursions.
                    # Use per-coordinate scaling so that relative movement is consistent.
                    scale_vec = step_size * (width_safe / np.max(width_safe))
                    noise = rng.normal(0.0, 1.0, dim) * scale_vec
                    x = x_best + noise

                # Boundary handling
                x = np.clip(x, lb, ub)
                y = eval_obj(x)

                if y < best_local_y:
                    best_local_x, best_local_y = x, y
                    improved = True

            # ---- Update global best ----
            if best_local_y < y_best:
                x_best, y_best = best_local_x, best_local_y

            # ---- Optional coordinate-wise refinement ----
            # If we haven't improved recently or step_size is still reasonably large,
            # scan a subset of coordinates around x_best with 1D moves.
            remaining = budget - evals_used
            if remaining <= 0:
                break

            # Decide to refine when either we just improved (to exploit) or
            # we're stalling (to escape shallow traps).
            # The scan is budget-aware and uses at most base_axes_attempt axes.
            do_refine = (improved or (outer % 3 == 2)) and (step_size > min_step)

            if do_refine and dim > 0:
                # Budget check for coordinate scan
                axes = min(dim, base_axes_attempt)
                # Choose a subset of axes, biased toward those with non-zero width
                nonzero = np.where(width > 0)[0]
                if nonzero.size > 0:
                    axes_idx = nonzero
                else:
                    axes_idx = np.arange(dim)

                if axes_idx.size > axes:
                    chosen = rng.choice(axes_idx, size=axes, replace=False)
                else:
                    chosen = axes_idx

                # Determine how many candidate evaluations we can afford for this refinement.
                # Each axis scan uses (2 * coord_scan_steps) evaluations (both directions).
                # We use no extra baseline because x_best already exists.
                est_cost = chosen.size * (2 * coord_scan_steps)
                if est_cost <= (budget - evals_used):
                    # 1D scan around x_best: step along coordinate +/-delta
                    # We shrink delta as coordinate scan proceeds to allow finer moves.
                    # If width is tiny for an axis, deltas are clipped to bounds anyway.
                    for ax in chosen:
                        if evals_used >= budget:
                            break
                        # Use local delta tied to step_size and the axis width
                        local_width = width_safe[ax]
                        base_delta = 0.5 * step_size * (local_width / np.max(width_safe))
                        if base_delta < min_step:
                            continue
                        for k in range(1, coord_scan_steps + 1):
                            if evals_used >= budget:
                                break
                            # Try +delta and -delta (symmetry)
                            delta = base_delta * (k / coord_scan_steps)
                            # plus
                            x = x_best.copy()
                            x[ax] = np.clip(x[ax] + delta, lb[ax], ub[ax])
                            y = eval_obj(x)
                            if y < y_best:
                                x_best, y_best = x, y
                                improved = True
                            # minus
                            x = x_best.copy()
                            x[ax] = np.clip(x[ax] - delta, lb[ax], ub[ax])
                            y = eval_obj(x)
                            if y < y_best:
                                x_best, y_best = x, y
                                improved = True

            # ---- Adapt step size ----
            # If we improved, enlarge slightly; otherwise shrink.
            # Also shrink over time to encourage convergence.
            shrink = 0.9 - 0.2 * t  # between 0.9 and 0.7
            grow = 1.05 + 0.1 * (1.0 - t)

            if improved:
                step_size = max(min_step, step_size * grow)
            else:
                step_size = max(min_step, step_size * shrink)

            # If step_size is too small relative to bounds, refresh with a random sample
            # (if budget remains) to avoid premature stagnation.
            if step_size <= min_step * 10 and (budget - evals_used) > 0:
                if rng.random() < 0.5:
                    x = lb + rng.random(dim) * (ub - lb)
                    y = eval_obj(x)
                    if y < y_best:
                        x_best, y_best = x, y
                    step_size = 0.25 * np.max(width_safe)

        return x_best, y_best
