# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, robust black-box minimization
# algorithm using a mix of global random search and local derivative-free
# refinement. It is designed for arbitrary dimensionality and works with
# unknown functions via only function evaluations.
# Search state: The algorithm maintains a current best point x_best and its
# objective value y_best, plus an iteration counter tracked against a hard
# evaluation budget.
# Candidate generation: Each cycle samples a small batch of candidate points.
# Candidates are drawn from (1) a “global” uniform distribution over the
# bounds and (2) a “local” Gaussian neighborhood around the current best.
# Local proposals are also biased using coordinate-wise perturbations to
# escape shallow flat regions.
# Selection and replacement: Among evaluated candidates, the best one that
# improves y_best (strictly) is adopted as the new x_best. Even if no
# improvement is found, the algorithm still continues with reduced local
# step sizes.
# Adaptation: The local neighborhood step size shrinks when improvements
# are hard to find, and it grows slightly when improvements occur.
# Exploration mechanisms: Random global sampling ensures broad coverage early
# in the budget and helps avoid being trapped in local minima.
# Exploitation mechanisms: Gaussian steps and coordinate-wise perturbations
# around x_best focus evaluations near the current best for refinement.
# Boundary handling: All candidate points are clamped to the provided bounds.
# If bounds are degenerate in a coordinate (lower==upper), that coordinate is
# kept fixed.
# Budget strategy: A strict evaluation counter is used; the algorithm never
# exceeds the provided budget. The batch size is chosen so the final batch
# fits.
# Closest known influences: The design is inspired by simple evolution-strategy
# style search (best-centered sampling) combined with stochastic global
# restarts and step-size adaptation, but implemented as a small standalone
# routine without external dependencies beyond numpy.
# Novelty or unusual aspects: The method includes a lightweight coordinate-wise
# perturbation proposal in the local neighborhood and a conservative step-size
# adaptation that is driven by observed improvements only.
# Failure modes: On extremely rugged landscapes or in the presence of severe
# noise, progress may stall; step sizes may shrink too quickly, leading to
# slow recovery. In very high dimensions, pure random global sampling may be
# inefficient, but local refinement helps when the initial best point is
# reasonably good.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # ---- Read bounds ----
        lb = None
        ub = None

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = func.lower
            ub = func.upper
        elif hasattr(func, "bounds"):
            b = func.bounds
            # support typical attributes: lb/ub
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = b.lb
                ub = b.ub
            elif hasattr(b, "lower") and hasattr(b, "upper"):
                lb = b.lower
                ub = b.upper

        if lb is None or ub is None:
            raise AttributeError(
                "Could not read bounds from func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)

        if lb.size != self.dim or ub.size != self.dim:
            raise ValueError(
                f"Bounds dimension mismatch: expected dim={self.dim}, got lb={lb.size}, ub={ub.size}."
            )

        # Handle degenerate dimensions (fixed coordinates).
        fixed = (lb == ub)
        span = ub - lb
        span_safe = np.where(fixed, 1.0, span)

        def clamp(x):
            # Fast clamp with fixed dims preserved.
            x = np.asarray(x, dtype=float)
            if np.any(fixed):
                # Clamp all, then force fixed to exact values.
                x = np.minimum(np.maximum(x, lb), ub)
                x[fixed] = lb[fixed]
                return x
            return np.minimum(np.maximum(x, lb), ub)

        # ---- Evaluation budget control ----
        max_evals = max(1, self.budget)
        evals = 0

        def eval_f(x):
            nonlocal evals
            if evals >= max_evals:
                # Never exceed the provided budget.
                return np.inf
            y = func(x)
            evals += 1
            # Ensure scalar float
            return float(np.asarray(y).reshape(()))

        # ---- Initialization: quick global sampling + set best ----
        # Batch size: adapt to budget to avoid overrun.
        # Start with up to 8 initial points or as many as budget allows.
        init_n = min(8, max_evals)
        # Choose a reasonable initial scale based on bounds.
        # Use average non-zero span as scale; fallback to 1.0.
        nonzero_spans = span[np.isfinite(span) & (span != 0)]
        base_scale = float(np.mean(np.abs(nonzero_spans))) if nonzero_spans.size else 1.0
        if not np.isfinite(base_scale) or base_scale <= 0:
            base_scale = 1.0

        # Global candidates uniformly within bounds.
        # For fixed dims, uniform sampling will collapse anyway, but clamp keeps exact values.
        X = lb + np.random.rand(init_n, self.dim) * span_safe
        X = np.array([clamp(x) for x in X], dtype=float)

        best_x = X[0]
        best_y = eval_f(best_x)
        for i in range(1, init_n):
            y = eval_f(X[i])
            if y < best_y:
                best_y = y
                best_x = X[i]

        # Local step size: starts relatively large but bounded.
        # Use a fraction of span to adapt to scale of domain.
        sigma = 0.35 * base_scale
        sigma_min = 1e-12 * base_scale
        sigma_max = 0.8 * base_scale

        # Coordinate-wise perturbation size.
        coord_step = 0.25 * base_scale

        # ---- Main loop: alternating global + local exploitation ----
        # We keep the per-iteration batch small to remain responsive to budget.
        # Approximately 8-20 iterations depending on budget.
        while evals < max_evals:
            remaining = max_evals - evals

            # Batch size: depends on remaining budget and dimension.
            # Smaller batch for very large dims to reduce overhead.
            if self.dim <= 10:
                batch = min(12, remaining)
            elif self.dim <= 50:
                batch = min(10, remaining)
            else:
                batch = min(8, remaining)

            # Decide exploration vs exploitation ratio.
            # Early: more global. Late: more local.
            progress = evals / max_evals
            global_frac = float(np.clip(0.65 - 0.55 * progress, 0.15, 0.65))
            n_global = int(round(batch * global_frac))
            n_global = min(batch, max(0, n_global))
            n_local = batch - n_global

            candidates = []

            # --- Global sampling candidates ---
            if n_global > 0:
                # Uniform across bounds with slight bias away from edges:
                # create by sampling in [lb+eps, ub-eps] where possible.
                eps = 1e-9 * base_scale
                lo = np.where(fixed, lb, lb + eps)
                hi = np.where(fixed, ub, ub - eps)
                # If lo>hi due to tiny spans, revert to exact bounds for those coords.
                lo = np.minimum(lo, hi)
                Xg = lo + np.random.rand(n_global, self.dim) * np.where(fixed, 0.0, (hi - lo))
                # Clamp to be safe and exact on fixed coords.
                for x in Xg:
                    candidates.append(clamp(x))

            # --- Local sampling candidates around best ---
            if n_local > 0:
                # Use best-centered Gaussian proposals; for fixed dims set to best values.
                # To reduce correlation and encourage coverage, draw with diagonal covariance.
                # Also add a coordinate-wise perturbation candidate each few samples.
                Xb = np.empty((n_local, self.dim), dtype=float)
                # Gaussian steps scaled by sigma and normalized per coordinate span.
                # Normalize so that if a coordinate has tiny span, steps are also tiny.
                span_scale = np.where(fixed, 0.0, span_safe)
                # Avoid division by zero: span_safe already set to 1 where fixed.
                # Use relative scaling: step ~ sigma * (span_i / avg_span)
                rel = span_scale / max(1e-12, base_scale)
                rel = np.where(fixed, 0.0, rel)

                # Create Gaussian candidates
                for j in range(n_local):
                    z = np.random.randn(self.dim)
                    step = sigma * rel * z
                    x = best_x + step
                    candidates.append(clamp(x))
                    Xb[j] = candidates[-1]

                # Coordinate-wise perturbations (small chance) to escape shallow plateaus.
                # Insert a few additional modifications if budget allows; otherwise they will
                # still be among 'candidates' already.
                # We'll overwrite some candidates with coordinate perturbations.
                k = min(2, n_local)  # at most 2 perturbations per batch
                if k > 0 and n_local >= 1:
                    idxs = np.random.choice(n_local, size=k, replace=False)
                    for idx in idxs:
                        x = np.array(candidates[self._global_count(candidates, idx)], dtype=float)
                        # Choose a coordinate with non-zero span.
                        free_coords = np.where(~fixed)[0]
                        if free_coords.size > 0:
                            c = int(np.random.choice(free_coords))
                            # Random sign and random magnitude between half and full coord_step
                            mag = coord_step * (0.5 + 0.5 * np.random.rand())
                            sign = 1.0 if np.random.rand() < 0.5 else -1.0
                            x[c] = x[c] + sign * mag
                            candidates[idx] = clamp(x)

            # ---- Evaluate and select best among candidates ----
            improved = False
            for x in candidates:
                if evals >= max_evals:
                    break
                y = eval_f(x)
                if y < best_y:
                    best_y = y
                    best_x = np.array(x, dtype=float)
                    improved = True

            # ---- Adaptation of local step size ----
            # Shrink if no improvement; gently expand if improved to keep exploration.
            if improved:
                sigma = min(sigma_max, sigma * 1.12)
                coord_step = min(base_scale, coord_step * 1.05)
            else:
                sigma = max(sigma_min, sigma * 0.82)
                coord_step = max(sigma_min, coord_step * 0.88)

            # Optional: small perturbation of best_x (restart-like) if stuck for long.
            # Keep it subtle to remain within budget constraints.
            # (We do not track "stuck time" across calls; just react to no-improvement batches.)
            if not improved and evals < max_evals:
                # One micro-restart attempt: sample near center of bounds.
                remaining = max_evals - evals
                if remaining > 0:
                    center = 0.5 * (lb + ub)
                    # Local around center with moderate sigma to diversify.
                    x = clamp(center + (0.3 * sigma) * np.random.randn(self.dim))
                    y = eval_f(x)
                    if y < best_y:
                        best_y = y
                        best_x = x
                        sigma = min(sigma_max, sigma * 1.05)

        return np.array(best_x, dtype=float), float(best_y)

    @staticmethod
    def _global_count(candidates, idx):
        # Helper to satisfy earlier coordinate-wise overwrite logic.
        # It returns idx itself; kept for backward compatibility with older refactors.
        # (No reliance on this function for correctness beyond providing a valid index.)
        return idx
