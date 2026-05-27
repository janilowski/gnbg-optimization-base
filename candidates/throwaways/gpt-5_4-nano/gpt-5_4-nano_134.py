# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm using a
# simple population-based evolutionary search with coordinate-wise local refinement.
# It works for any dimension >= 1, respects the provided evaluation budget, and only
# relies on bounds obtained from the given objective function.
#
# Search state: Maintains a small population of candidate solutions (a center plus
# mutated points). Keeps track of the current best solution found (best_x, best_y).
#
# Candidate generation: Each iteration creates offspring by applying isotropic and
# coordinate-skewed Gaussian mutations around the current center. Mutation scales
# shrink over time to gradually shift from exploration to exploitation.
#
# Selection and replacement: Offspring are evaluated and the best among the current
# center and offspring becomes the new center; additionally, a few elites can be
# retained implicitly by comparing only against the current best.
#
# Adaptation: The mutation step size is adapted based on progress: it shrinks
# when improvements happen (to exploit) and grows slightly when stuck (to escape).
# A lightweight coordinate-wise local search is applied near the end using multiple
# 1D perturbation probes around the best point.
#
# Exploration mechanisms: Early iterations use larger mutation scales and multiple
# offspring per iteration with randomness and occasional coordinate-targeted moves.
#
# Exploitation mechanisms: Later iterations reduce mutation variance and perform
# coordinate-wise probing around the best known point.
#
# Boundary handling: All candidate points are clipped to the feasible bounds after
# mutation; clipping ensures evaluations always satisfy constraints.
#
# Budget strategy: The algorithm tracks the exact number of objective evaluations and
# never exceeds the provided budget. It stops early if the budget would be exceeded.
#
# Closest known influences: The design resembles a minimal evolutionary strategy (ES)
# / CMA-lite spirit (population mutations + step-size adaptation), combined with a
# deterministic coordinate probing refinement.
#
# Novelty or unusual aspects: Uses a hybrid approach that mixes isotropic Gaussian
# mutations with coordinate-targeted offspring and a bounded coordinate-wise local
# refinement, all while keeping the implementation short and dimension-agnostic.
#
# Failure modes: In very rugged landscapes or extremely tight bounds, clipping can
# lead to many similar samples and slow progress. With too small budgets, the
# algorithm may only perform coarse global search without enough refinement.
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
        if dim <= 0:
            raise ValueError("dim must be >= 1")
        if budget <= 0:
            raise ValueError("budget must be >= 1")

        # ---- Read bounds from func ----
        lb = ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and func.bounds is not None:
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
        if lb is None or ub is None:
            raise AttributeError(
                "Objective function must provide bounds via func.lower/func.upper "
                "or func.bounds.lb/func.bounds.ub."
            )
        if lb.shape == () and dim > 1:
            lb = np.full(dim, float(lb))
        if ub.shape == () and dim > 1:
            ub = np.full(dim, float(ub))
        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Ensure valid bounds ordering
        if np.any(ub < lb):
            raise ValueError("Invalid bounds: found ub < lb for some dimensions.")

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        # Small helper: evaluate and account for budget
        evals = 0
        best_x = None
        best_y = None

        def evaluate(x):
            nonlocal evals, best_x, best_y
            if evals >= budget:
                return None
            x = clip(np.asarray(x, dtype=float))
            y = float(func(x))
            evals += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = x
            return y

        # ---- Initialization ----
        # Use mid-point as a deterministic start, plus a few random points.
        mid = 0.5 * (lb + ub)

        # If bounds are effectively a point in all dims, just evaluate once.
        span = ub - lb
        typical_span = float(np.linalg.norm(span) / np.sqrt(dim)) if dim > 0 else 1.0
        typical_span = typical_span if typical_span > 0 else 1.0

        # Set initial step size relative to box size
        step = 0.25 * typical_span
        step = max(step, 1e-12)

        # Budget for initial sampling
        # Keep it small to leave room for iterative progress
        init_tries = min(3 * dim, max(1, budget // 10))
        evaluate(mid)
        # Use additional random points for diversity
        for _ in range(max(0, init_tries - 1)):
            if evals >= budget:
                break
            r = np.random.random(dim)
            x = lb + r * (ub - lb)
            evaluate(x)

        if evals >= budget:
            return best_x, best_y

        # ---- Evolutionary loop ----
        # Number of iterations chosen so we stay within the budget.
        # Each iteration evaluates `offspring_k` points.
        # Keep offspring small in high dimension.
        # Allocate roughly 70% budget to evolutionary search, 30% to refinement.
        evo_budget = int(max(1, 0.7 * budget))
        ref_budget = budget - evo_budget
        if ref_budget < 0:
            ref_budget = 0

        # Determine iterations by remaining evaluations
        offspring_k = int(max(2, min(12, 2 + dim // 2)))
        # Ensure not exceeding evo_budget
        # iterations such that evals + iterations*offspring_k <= evo_budget
        remaining_evo = max(0, evo_budget - evals)
        if remaining_evo <= 0:
            offspring_k = 0
        iters = remaining_evo // offspring_k if offspring_k > 0 else 0
        if iters <= 0 and offspring_k > 0 and evals < evo_budget:
            iters = 1

        center = np.array(best_x, copy=True)
        center_y = float(best_y)

        # Progress tracking for step-size adaptation
        no_improve = 0
        last_best_y = center_y

        for _ in range(iters):
            if evals >= evo_budget or evals >= budget:
                break

            improved = False
            # Slightly shrink step each iteration; will also be adjusted by adaptation below.
            # Use a nonlinear decay to keep exploration at start.
            decay = 0.90 + 0.10 * (evals / max(1.0, evo_budget))
            local_step = step * decay

            # Generate offspring; some are isotropic, some are coordinate-skewed.
            # We always clip to bounds.
            for j in range(offspring_k):
                if evals >= evo_budget or evals >= budget:
                    break
                z = np.random.normal(size=dim)

                # Occasionally bias mutation towards a randomly chosen coordinate.
                if dim >= 2 and (j % 3 == 2):
                    k = np.random.randint(dim)
                    z = np.zeros(dim)
                    z[k] = np.random.normal()

                # Scale mutations: use both isotropic and per-coordinate factors
                # to maintain robustness when different coordinate spans exist.
                span_safe = np.where(span > 0, span, 1.0)
                # Per-coordinate scale based on relative span.
                rel = span_safe / max(1e-12, float(np.max(span_safe)))
                scale_vec = rel ** 0.5  # compress extremes a bit

                x = center + local_step * scale_vec * z
                y = evaluate(x)
                if y is not None and y < center_y - 1e-16:
                    center_y = y
                    center = np.array(best_x, copy=True)
                    improved = True

            # Adaptation
            if improved:
                no_improve = 0
                step *= 0.85
            else:
                no_improve += 1
                # If stuck, re-expand slightly but not too much.
                if no_improve >= 2:
                    step *= 1.08
                    no_improve = 0

            # Also ensure step doesn't collapse to zero (numerical/box tightness)
            min_step = 1e-12 * typical_span
            step = max(step, min_step)

            # Early exit if evo budget is reached
            if evals >= evo_budget:
                break

            # Keep center synced with global best in case of ties/rounding
            if best_y is not None and best_y < center_y:
                center_y = best_y
                center = np.array(best_x, copy=True)

        if evals >= budget:
            return best_x, best_y

        # ---- Coordinate-wise local refinement ----
        # Perform a small number of 1D probes around best_x.
        if ref_budget > 0 and evals < budget:
            # Remaining budget for refinement
            remaining = budget - evals
            # Use at most one probe set per coordinate, capped by remaining budget.
            # Each coordinate uses up to 2 evaluations (±delta).
            # If budget is tiny, will still try a few coordinates.
            max_coords = min(dim, max(1, remaining // 2))

            # Probe magnitude decreases with progress.
            frac = (evals / max(1, budget))
            probe_step = step * (0.5 - 0.3 * frac)
            probe_step = max(probe_step, 1e-12 * typical_span)

            # Choose coordinate order: prefer larger span dimensions.
            span_weights = np.where(span > 0, span, 1.0)
            order = np.argsort(-span_weights)  # descending
            order = order[:max_coords]

            x0 = np.array(best_x, copy=True)
            for k in order:
                if evals >= budget:
                    break
                # Evaluate +delta and -delta if budget allows
                delta = probe_step * (0.7 + 0.6 * np.random.random())
                cand_plus = x0.copy()
                cand_minus = x0.copy()
                cand_plus[k] = cand_plus[k] + delta
                cand_minus[k] = cand_minus[k] - delta

                if evals < budget:
                    evaluate(cand_plus)
                if evals < budget:
                    evaluate(cand_minus)

                # Update local reference to the best found so far
                if best_x is not None:
                    x0 = np.array(best_x, copy=True)

            # Final tiny isotropic exploitation step if budget remains
            while evals < budget and np.random.random() < 0.8:
                if budget - evals < 1:
                    break
                r = np.random.normal(size=dim)
                x = x0 + (0.35 * probe_step) * r
                evaluate(x)
                # Stop if step is no longer meaningful
                if probe_step <= 1e-12 * typical_span:
                    break

        return best_x, best_y
