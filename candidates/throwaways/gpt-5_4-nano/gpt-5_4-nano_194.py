# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy combining
# global random sampling with iterative local refinement using a coordinate-
# style “pattern” search around the current best. It is designed to be robust
# across dimensions while staying within a strict evaluation budget.
# Search state: Maintains the current best point x_best and its objective value
# y_best, plus a step size controlling the radius of local exploration.
# Candidate generation: Each iteration generates a small set of candidate
# points: (1) one random “global” proposal occasionally, and (2) several
# structured local proposals obtained by perturbing the best point along
# coordinate directions and also via a few isotropic Gaussian steps.
# Selection and replacement: Evaluates all candidates that fit in remaining
# budget, then keeps the best among them. If improvement occurs, the local
# step size is increased slightly; otherwise it is reduced.
# Adaptation: Step size adapts multiplicatively based on whether the search
# improves the incumbent solution, balancing exploration and exploitation.
# Exploration mechanisms: Uses random proposals (uniform within bounds and
# occasional Gaussian perturbations) early and sporadically later.
# Exploitation mechanisms: Uses coordinate/pattern perturbations around the
# incumbent to refine the best found region.
# Boundary handling: All candidate points are clipped to the provided bounds.
# Budget strategy: Tracks the number of objective evaluations and never exceeds
# the given evaluation budget; the loop terminates when the budget is exhausted.
# Closest known influences: Similar in spirit to evolutionary restarts and
# pattern/coordinate search with adaptive step sizes, but kept intentionally
# simple for black-box benchmarking.
# Novelty or unusual aspects: Combines coordinate-pattern moves with a small
# “best-of-batch” selection per iteration and adaptive step scaling that reacts
# to success/failure.
# Failure modes: If the objective is extremely noisy, non-smooth, or
# adversarial to coordinate moves, progress may be slow; if bounds are very
# tight, clipping can reduce effective search diversity.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # ---- Read bounds robustly from func ----
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            # support b.lb / b.ub (common in benchmark APIs)
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            else:
                raise AttributeError("func.bounds must have lb and ub.")
        else:
            raise AttributeError("func must have lower/upper or bounds.lb/bounds.ub.")

        lb = np.broadcast_to(lb, (self.dim,)).copy()
        ub = np.broadcast_to(ub, (self.dim,)).copy()
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            raise ValueError("Bounds must be finite.")
        if np.any(ub < lb):
            raise ValueError("Upper bounds must be >= lower bounds.")

        # Handle degenerate bounds: ensure step sizes become 0 where needed.
        span = ub - lb
        span_safe = np.where(span > 0, span, 1.0)

        # ---- Budget accounting and evaluation wrapper ----
        evals = 0
        max_evals = max(1, self.budget)

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        def eval_one(x):
            nonlocal evals
            if evals >= max_evals:
                # Should never happen; keep as a hard guard.
                return np.inf
            x = np.asarray(x, dtype=float).reshape(-1)
            y = func(x)
            evals += 1
            # Expect scalar; coerce robustly
            return float(np.asarray(y).reshape(()))

        # ---- Initialization: random sampling for a strong starting point ----
        # We take a small initial design then switch to iterative refinement.
        # Ensure at least 1 evaluation.
        init_budget = min(max_evals, max(4, self.dim + 1))
        x_best = None
        y_best = np.inf

        for _ in range(init_budget):
            # Uniform in bounds
            r = np.random.random(self.dim)
            x = lb + r * span_safe
            x = np.where(span > 0, x, lb)  # exact for degenerate coords
            y = eval_one(clip(x))
            if y < y_best:
                y_best = y
                x_best = x.copy()

        # If budget is exhausted, return what we have.
        if evals >= max_evals:
            return x_best, y_best

        # ---- Step size: start as a fraction of span ----
        # Avoid zero step for degenerate dimensions.
        step = 0.25 * span_safe
        step = np.where(span > 0, step, 0.0)

        # Iteration count estimated from remaining budget with small batch size.
        # Keep small batch to reduce wasted evaluations when improvements are rare.
        batch_local = max(2, min(2 * self.dim + 1, 32))
        # We'll pick candidates per iteration: 1 global (optional) + pattern + gaussian.
        pattern_tries = min(self.dim, max(1, (batch_local - 1) // 2))
        gauss_tries = max(1, batch_local - 1 - pattern_tries)

        while evals < max_evals:
            remaining = max_evals - evals
            # Decide how many candidates we can evaluate this iteration.
            # Candidate set size:
            # - optional global: 1
            # - coordinate/pattern: pattern_tries
            # - gaussian: gauss_tries
            # Total <= batch_local and <= remaining.
            want_global = 1 if np.random.rand() < 0.25 else 0  # occasional exploration
            total_candidates = want_global + pattern_tries + gauss_tries
            total_candidates = min(total_candidates, remaining)

            if total_candidates <= 0:
                break

            candidates = []

            # Optional global candidate: sample around best with a broader spread or uniform.
            if want_global and len(candidates) < total_candidates:
                if np.random.rand() < 0.5:
                    # Uniform exploration
                    r = np.random.random(self.dim)
                    xg = lb + r * span_safe
                    xg = np.where(span > 0, xg, lb)
                else:
                    # Broader Gaussian around incumbent
                    sigma = max(1e-12, np.linalg.norm(step) / np.sqrt(self.dim)) if self.dim > 0 else 0.0
                    xg = x_best + np.random.randn(self.dim) * (0.8 * sigma)
                candidates.append(clip(xg))

            # Pattern / coordinate exploitation: perturb along several coordinates.
            # Choose random subset of coordinates for variety (more efficient than all).
            if pattern_tries > 0 and len(candidates) < total_candidates:
                # Determine how many pattern moves we can include this iteration.
                k_pat = min(pattern_tries, total_candidates - len(candidates))
                coords = np.random.choice(self.dim, size=k_pat, replace=False) if self.dim > 1 else np.array([0])
                # Alternate directions to get both + and - patterns when possible.
                # For each coord, try one direction; if we can, also try opposite direction.
                for i, c in enumerate(coords):
                    if len(candidates) >= total_candidates:
                        break
                    direction = 1.0 if (np.random.rand() < 0.5) else -1.0
                    x = x_best.copy()
                    x[c] = x_best[c] + direction * max(1e-12, step[c])
                    candidates.append(clip(x))

            # Gaussian exploitation: a few isotropic moves scaled by step magnitude.
            if len(candidates) < total_candidates:
                k_gauss = min(gauss_tries, total_candidates - len(candidates))
                # Use a representative scale from step.
                scale = np.linalg.norm(step) / np.sqrt(self.dim) if self.dim > 0 else 0.0
                scale = max(scale, 1e-12)
                for _ in range(k_gauss):
                    x = x_best + np.random.randn(self.dim) * scale
                    candidates.append(clip(x))

            # Evaluate batch and select best (minimization)
            y_batch_best = y_best
            x_batch_best = x_best

            for x in candidates:
                y = eval_one(x)
                if y < y_batch_best:
                    y_batch_best = y
                    x_batch_best = np.asarray(x, dtype=float).copy()

            # If improved, update incumbent and increase step slightly; else shrink.
            if y_batch_best < y_best - 1e-15:
                x_best = x_batch_best
                y_best = y_batch_best
                # Gentle expansion to capitalize on improved region.
                step = np.where(step > 0, step * 1.08, 0.0)
            else:
                # Shrink step to focus search locally.
                step = np.where(step > 0, step * 0.82, 0.0)

            # If step becomes tiny across all dimensions, we can still continue
            # with occasional random proposals until budget ends.
            if evals >= max_evals:
                break

        return x_best, y_best
