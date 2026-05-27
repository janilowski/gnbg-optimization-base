# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, robust black-box minimization strategy
# using a population-free (single-point) trust-region style search with stochastic
# global restarts. It combines exploitation around the current best point with
# occasional diversification to escape local minima.
# Search state: The algorithm maintains a current best point x_best and its objective
# value y_best, plus a scalar step size (sigma) that controls how far new candidates
# are sampled from x_best.
# Candidate generation: Each iteration samples a small set of candidate points from
# a Gaussian distribution centered at x_best with scale sigma. It also includes
# one purely random candidate every few iterations (or upon stagnation).
# Selection and replacement: The best candidate among those evaluated becomes the new
# x_best if it improves y_best; otherwise the search may contract sigma.
# Adaptation: If no improvement occurs for a few evaluations, sigma is reduced to focus
# search locally; if improvement is found, sigma is mildly expanded to encourage progress.
# Exploration mechanisms: Random restart/diversification injects uniformly sampled points
# (or re-centering after stagnation) to explore new regions.
# Exploitation mechanisms: Local sampling around x_best with decreasing sigma forms the
# exploitation behavior.
# Boundary handling: Candidates are clipped to the provided box bounds and then
# evaluated; this keeps all points feasible.
# Budget strategy: The implementation strictly caps the number of objective evaluations
# to the provided budget by tracking an internal evaluation counter and selecting
# candidate counts per iteration that never exceed remaining budget.
# Closest known influences: The design is inspired by simple evolution-strategy/trust-region
# heuristics (1+lambda) with adaptive step-size and occasional restarts.
# Novelty or unusual aspects: It uses a small, dimension-robust candidate batch size
# derived from the dimension (bounded to keep runtime stable) and a straightforward
# stagnation counter to trigger sigma shrinkage and diversification.
# Failure modes: If the objective is extremely noisy or highly deceptive, sigma adaptation
# may oscillate; excessive shrinking can slow progress until a restart happens.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import math
import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0 or dim <= 0:
            # With no budget, we cannot evaluate; return a benign default.
            return np.zeros(dim, dtype=float), float("inf")

        # ---- Read bounds from func ----
        lower = None
        upper = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = func.lower
            upper = func.upper
        elif hasattr(func, "bounds"):
            b = func.bounds
            # Accept either .lb/.ub or similar names
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lower = b.lb
                upper = b.ub

        if lower is None or upper is None:
            # No bounds: still required by prompt to read; fall back to wide box.
            lower = np.full(dim, -1.0, dtype=float)
            upper = np.full(dim, 1.0, dtype=float)

        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        if lower.size != dim or upper.size != dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Ensure well-formed bounds
        lo = np.minimum(lower, upper)
        hi = np.maximum(lower, upper)

        # Vectorized clipping
        def clip_to_bounds(x):
            return np.minimum(np.maximum(x, lo), hi)

        # ---- Evaluation wrapper to strictly respect the budget ----
        eval_count = 0

        def eval_obj(x):
            nonlocal eval_count
            if eval_count >= budget:
                # Never exceed budget; return a large value to avoid changing best.
                return float("inf")
            y = func(x)
            eval_count += 1
            return float(y)

        # ---- Initialize: evaluate one random point ----
        rng = np.random
        # Use uniform sampling within bounds for initial point
        x_best = lo + rng.rand(dim) * (hi - lo)
        y_best = eval_obj(x_best)

        # Initialize sigma based on box size (robust across dimensions)
        box_span = hi - lo
        span_scale = float(np.mean(box_span)) if dim > 0 else 1.0
        # Avoid zero span
        if not np.isfinite(span_scale) or span_scale <= 0:
            span_scale = 1.0
        sigma = 0.3 * span_scale

        # Candidate batch size: dimension-robust and bounded
        # Larger dims get smaller batches to reduce expensive evaluations.
        lam = int(max(2, min(12, round(4 + 0.5 * math.sqrt(dim)))))
        # Add extra random candidate sometimes
        random_every = int(max(3, min(10, round(5 + dim * 0.1))))

        # Stagnation control
        no_improve = 0
        stagnation_limit = int(max(4, min(20, round(6 + 0.2 * dim))))

        # Track total iterations roughly by batches (evaluation-driven loop)
        while eval_count < budget:
            # Determine remaining budget and how many to evaluate this iteration
            rem = budget - eval_count
            batch = min(lam, rem)

            # Optional diversification (one random point) if budget allows and at interval
            include_random = (eval_count < budget) and (no_improve >= stagnation_limit or (eval_count % (random_every * lam)) == 0)
            if include_random and rem >= batch + 1:
                cand_total = batch + 1
                batch_random_pos = True
            else:
                cand_total = batch
                batch_random_pos = False

            # ---- Candidate generation ----
            # Start with local samples around best.
            # Use isotropic Gaussian steps; enforce feasibility via clipping.
            # Use an orthogonal-ish scaling via random direction normalization to avoid
            # extremely large outliers dominating too often.
            # Still clipped to box.
            candidates = []
            for _ in range(batch):
                z = rng.randn(dim)
                # Normalize direction to keep step magnitude mostly controlled
                norm = float(np.linalg.norm(z))
                if norm > 0:
                    z = z / norm
                step = sigma * z
                x = clip_to_bounds(x_best + step)
                candidates.append(x)

            if batch_random_pos:
                x_rand = lo + rng.rand(dim) * (hi - lo)
                candidates.append(x_rand)

            # ---- Selection and replacement ----
            y_local_best = y_best
            x_local_best = x_best

            for x in candidates:
                # In case we accidentally exceed budget due to rounding
                if eval_count >= budget:
                    break
                y = eval_obj(x)
                if y < y_local_best:
                    y_local_best = y
                    x_local_best = x

            # Update global best if improved
            improved = y_local_best < y_best
            if improved:
                x_best, y_best = x_local_best, y_local_best
                no_improve = 0
                # Mildly increase sigma to keep exploring around new basin
                sigma *= 1.08
            else:
                no_improve += 1
                # Shrink sigma to focus locally
                sigma *= 0.82

            # ---- Adaptation / exploration triggers ----
            # If stagnation persists, diversify: re-center at best but sample wider
            # or perform a restart point evaluation (only if budget allows).
            if no_improve >= stagnation_limit and eval_count < budget:
                # Restart: evaluate a few wide random points but never exceed budget.
                # Keep it minimal (1 point) to remain compact.
                # If sigma is already tiny, ensure a meaningful jump by enlarging.
                sigma = max(sigma, 0.5 * span_scale)
                rem = budget - eval_count
                if rem > 0:
                    x_restart = lo + rng.rand(dim) * (hi - lo)
                    y_restart = eval_obj(x_restart)
                    if y_restart < y_best:
                        x_best, y_best = x_restart, y_restart
                        no_improve = 0
                    else:
                        no_improve = max(0, no_improve - 1)
                # Continue loop; sigma will continue adapting.

            # Safety: prevent sigma from collapsing completely
            # Use a lower bound relative to box span to avoid total stagnation.
            lower_sigma = 1e-12 * (span_scale if span_scale > 0 else 1.0)
            sigma = float(max(sigma, lower_sigma))
            # Also cap sigma to avoid overly huge steps in bounded problems
            sigma = float(min(sigma, 2.0 * span_scale + 1e-12))

        return np.asarray(x_best, dtype=float), float(y_best)
