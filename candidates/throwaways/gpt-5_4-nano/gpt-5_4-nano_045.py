# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy using
# coordinate-wise Gaussian perturbations combined with a simple global
# memory/restarts scheme. It tracks the best point found and refines it
# using decreasing step sizes while periodically exploring from that best.
# Search state: Maintains current center x, best point best_x/best_y,
# current step scale sigma, remaining evaluation budget, and a small set
# of candidate points sampled around the center.
# Candidate generation: At each iteration, samples a batch of points by
# adding Gaussian noise scaled by sigma to the current center. Also uses
# occasional coordinate-level perturbations to improve robustness on
# axis-aligned structures.
# Selection and replacement: Evaluates each candidate, updates best_x/best_y,
# then selects the best among batch (ties by earlier occurrence) as the
# next center to tighten the search.
# Adaptation: Uses a success rule: if an improvement is found, sigma is
# slightly reduced (finer exploitation); otherwise sigma is enlarged
# (encourages escape). Step size decays slowly with the iteration count.
# Exploration mechanisms: Random restarts from the current best with larger
# sigma when stagnation is detected; also occasional coordinate perturbations.
# Exploitation mechanisms: Greedy replacement by the best batch point and
# shrinking sigma after improvements.
# Boundary handling: Clamps candidate points to the provided bounds after
# perturbations. Bounds are read from func.lower/upper or func.bounds.lb/ub.
# Budget strategy: Never exceeds the provided evaluation budget by allocating
# a fixed budget per stage and then stopping immediately when evaluations
# reach the limit.
# Closest known influences: Inspired by evolution strategies / CMA-like
# step-size control concepts, but kept intentionally minimal and
# dimension-robust for a benchmarking harness.
# Novelty or unusual aspects: Uses a hybrid of Gaussian batch sampling and
# a light coordinate perturbation heuristic, along with a small restart
# logic tied to stagnation.
# Failure modes: If the objective has extremely sharp features or highly
# deceptive landscapes, the simple adaptation may under-explore or converge
# prematurely. Clamping can also flatten gradients near boundaries.
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
        if budget <= 0:
            # No evaluations possible; return a zero vector as a placeholder.
            return np.zeros(dim, dtype=float), float("inf")

        # ---- Read bounds ----
        # Accepts:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        lower = upper = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lower = np.asarray(func.bounds.lb, dtype=float)
            upper = np.asarray(func.bounds.ub, dtype=float)
        else:
            # If bounds are not provided, assume a generic box.
            # (Benchmark should supply bounds; this is a fallback.)
            lower = -np.ones(dim, dtype=float)
            upper = np.ones(dim, dtype=float)

        # Ensure correct shape
        lower = np.broadcast_to(lower, (dim,)).astype(float, copy=False)
        upper = np.broadcast_to(upper, (dim,)).astype(float, copy=False)
        # Handle degenerate bounds safely
        span = upper - lower
        span = np.where(span == 0.0, 1.0, span)

        def clamp(x):
            return np.minimum(upper, np.maximum(lower, x))

        # ---- Evaluation wrapper with strict budget accounting ----
        evals = 0

        def evaluate(x):
            nonlocal evals
            if evals >= budget:
                # Do not call beyond the budget.
                return float("inf")
            x = clamp(np.asarray(x, dtype=float))
            y = float(func(x))
            evals += 1
            return y

        # ---- Initialization ----
        # Start from a random point uniformly within bounds.
        x = lower + (upper - lower) * np.random.rand(dim)
        x = clamp(x)
        y = evaluate(x)

        best_x = x.copy()
        best_y = y

        # Step scale: use fraction of box size.
        # Start somewhat aggressive but robust.
        sigma0 = 0.25 * np.mean(np.abs(span))
        if not np.isfinite(sigma0) or sigma0 <= 0:
            sigma0 = 1.0

        sigma = sigma0
        # Stagnation counter to decide restarts.
        no_improve = 0

        # Helper to compute remaining evaluations
        def remaining():
            return budget - evals

        # ---- Main loop ----
        # Iterations: we don't assume a fixed number of generations; we use remaining budget.
        # Batch size chosen based on remaining budget to reduce overhead.
        while remaining() > 0:
            # Determine batch size (1..min(10, remaining))
            # Larger batches explore more; keep small for compactness.
            bmax = min(10, remaining())
            # Ensure at least 2 candidates if possible to allow improvement selection.
            batch_size = 2 if remaining() >= 2 else 1
            batch_size = min(batch_size, bmax)

            # Slow decay of sigma to improve late-stage refinement
            # but allow increases when stagnating.
            progress = evals / max(1, budget)
            sigma *= (1.0 - 0.15 / max(1.0, 1.0 + 4.0 * progress))

            # Candidate generation around current center (best_x as exploitation anchor)
            center = best_x.copy()

            # Gaussian samples
            # Shape: (batch_size, dim)
            noise = np.random.randn(batch_size, dim)
            # Scale per-dimension proportional to bounds span to keep units consistent.
            step = sigma * (span / max(1e-12, np.mean(np.abs(span))))
            candidates = center[None, :] + noise * step[None, :]

            # Occasional coordinate perturbation to improve axis-aligned performance.
            # Do it for one extra candidate if budget allows.
            coord_extra = False
            if remaining() - batch_size >= 1 and (no_improve >= 2 or np.random.rand() < 0.3):
                coord_extra = True

            if coord_extra:
                # Add one coordinate-perturbed point
                k = np.random.randint(0, dim)
                c = center.copy()
                # Sign random, magnitude from normal distribution
                c[k] = c[k] + np.random.randn() * (0.75 * sigma * (span[k] / max(1e-12, np.mean(np.abs(span)))))
                candidates = np.vstack([candidates, c[None, :]])
                b_eval = candidates.shape[0]
            else:
                b_eval = candidates.shape[0]

            # Ensure we don't exceed remaining evaluations
            if b_eval > remaining():
                candidates = candidates[: remaining(), :]
                b_eval = candidates.shape[0]

            # Selection: evaluate all candidates, pick best among them
            local_best_x = None
            local_best_y = float("inf")

            for i in range(b_eval):
                cx = candidates[i]
                cy = evaluate(cx)
                if cy < local_best_y:
                    local_best_y = cy
                    local_best_x = clamp(cx)

            # Update global best and adaptation
            improved = local_best_y < best_y
            if improved:
                best_y = local_best_y
                best_x = local_best_x
                no_improve = 0
                # Tighten exploitation when success occurs
                sigma *= 0.85
            else:
                no_improve += 1
                # Expand search radius when stagnating
                sigma *= 1.15 + 0.05 * no_improve

            # Restart mechanism: if stagnation persists, sample a fresh center near bounds.
            # We keep it budget-aware by doing at most one restart per loop when possible.
            if no_improve >= 4 and remaining() > 0:
                # Restart from a random point near the current best (or uniform if sigma too small)
                # Pick a random mixing weight.
                mix = 0.5 + 0.5 * np.random.rand()
                if sigma < 1e-14:
                    # Uniform restart
                    x_new = lower + (upper - lower) * np.random.rand(dim)
                else:
                    # Perturb best by a larger jump
                    jump = (1.5 * sigma) * (span / max(1e-12, np.mean(np.abs(span)))) * np.random.randn(dim)
                    x_new = best_x + jump
                x_new = clamp((1 - mix) * best_x + mix * x_new)
                y_new = evaluate(x_new)
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new
                    no_improve = 0
                    sigma *= 0.9
                else:
                    # If restart didn't help, continue with larger sigma to explore.
                    sigma *= 1.25

        return best_x, best_y
