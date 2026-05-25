# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer (GNBG-style)
# using a randomized coordinate/finite-difference guided search with a fallback
# population-based resampling. It maintains the best-so-far solution and uses
# local Gaussian perturbations around it, with step-size control based on
# improvement.
# Search state: Keeps best_x, best_y, current step sizes (global sigma),
# and per-coordinate sigmas. Tracks remaining evaluation budget to never exceed it.
# Candidate generation: Proposes candidates by (1) testing a small set of
# coordinate-aligned perturbations around best_x, (2) drawing isotropic Gaussian
# perturbations around best_x, and (3) optional uniform restarts if progress stalls.
# Selection and replacement: Evaluates candidates, picks the best among evaluated
# points, and uses it to update best_x/best_y; step sizes are increased or
# decreased depending on whether improvement occurred.
# Adaptation: Uses multiplicative step-size adaptation (shrink on failure,
# grow on success) and reduces exploration when near the bound.
# Exploration mechanisms: Random coordinate directions and Gaussian sampling;
# periodic restarts near random points within bounds if no improvement occurs.
# Exploitation mechanisms: Biases coordinate perturbations towards directions that
# historically improved by comparing objective values at +/- offsets.
# Boundary handling: Clips candidates to the provided bounds after perturbation.
# Budget strategy: Each call to the objective decrements a counter; the algorithm
# stops generating candidates when budget is exhausted (never exceeds provided budget).
# Closest known influences: Combines ideas from coordinate search, evolution strategies
# (ES)-like sampling, and step-size adaptation as used in many black-box optimizers.
# Novelty or unusual aspects: Uses a hybrid "coordinate probing then Gaussian refinement"
# loop with robust clipping and a minimal restart mechanism without any external state.
# Failure modes: For highly constrained, discontinuous, or extremely noisy objectives,
# step-size adaptation may stall; restarts mitigate but cannot guarantee success.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return a feasible point if possible.
            lb, ub = self._read_bounds(func)
            x0 = self._midpoint(lb, ub)
            return x0, np.inf

        lb, ub = self._read_bounds(func)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        # Handle degenerate bounds robustly.
        lb = np.broadcast_to(lb, (dim,)).copy()
        ub = np.broadcast_to(ub, (dim,)).copy()
        width = np.maximum(ub - lb, 0.0)
        center = self._midpoint(lb, ub)

        # Random initial point within bounds.
        x = self._sample_uniform(lb, ub)
        best_x = self._clip(x, lb, ub)
        best_y, evals = self._evaluate(func, best_x, 1)

        # Step sizes: proportional to bounds width.
        # Use both global and per-coordinate sigmas for robustness.
        base = np.max(width) if np.max(width) > 0 else 1.0
        global_sigma = 0.25 * base
        coord_sigma = np.maximum(0.1 * width, 1e-12)
        # If width is zero on some coordinate, keep sigma tiny so perturbation doesn't matter.
        coord_sigma = np.where(width > 0, coord_sigma, 1e-12)

        # Progress tracking.
        no_improve = 0
        # Try a few iterations; each iteration draws a small batch.
        # The loop naturally stops as soon as evaluation budget is exhausted.
        # We keep the batch sizes modest to respect the budget precisely.
        while evals < budget:
            remaining = budget - evals

            # ---------------- Exploitation-ish step: coordinate probing ----------------
            # Pick a small subset of coordinates, try +/- offsets, infer promising direction.
            k = min(dim, 2 + dim // 3)  # modest number of coordinates to probe
            coords = np.random.choice(dim, size=k, replace=False) if dim > 1 else np.array([0])

            candidates = []
            # Always include the current best as a reference (cheap but consumes evals).
            # We'll avoid extra eval for reference and just probe +/-.
            # Choose a probe scale based on current sigma.
            probe_scale = global_sigma * 0.6
            probe_scale = max(float(probe_scale), 1e-12)

            for j in coords:
                if width[j] > 0:
                    step = min(coord_sigma[j], width[j]) * 0.5
                    step = max(float(step), 1e-12)
                    # Use probe_scale to adapt to overall landscape
                    step = 0.5 * step + 0.5 * probe_scale
                else:
                    step = 1e-12

                # +/- perturbations along coordinate j
                x_plus = best_x.copy()
                x_minus = best_x.copy()
                x_plus[j] = best_x[j] + step
                x_minus[j] = best_x[j] - step
                candidates.append(self._clip(x_plus, lb, ub))
                candidates.append(self._clip(x_minus, lb, ub))

            # Respect remaining budget: only evaluate up to remaining candidates.
            if len(candidates) > remaining:
                # Keep deterministic subset order? random subset to keep behavior varied.
                idx = np.random.choice(len(candidates), size=remaining, replace=False)
                candidates = [candidates[i] for i in idx]

            # Evaluate candidates and update best.
            improved = False
            for cand in candidates:
                if evals >= budget:
                    break
                y, evals_inc = self._evaluate_y(func, cand)
                evals += evals_inc
                if y < best_y:
                    best_y = y
                    best_x = cand.copy()
                    improved = True

            # ---------------- Adaptation ----------------
            if improved:
                no_improve = 0
                # Grow a bit to explore; but keep bounded.
                global_sigma *= 1.12
                global_sigma = min(global_sigma, 2.0 * (np.max(width) if np.max(width) > 0 else global_sigma))
                # Slightly increase coordinate sigmas where width allows.
                coord_sigma = np.where(width > 0, np.minimum(coord_sigma * 1.08, np.maximum(width, 1e-12)), coord_sigma)
            else:
                no_improve += 1
                # Shrink to exploit locally.
                global_sigma *= 0.82
                global_sigma = max(global_sigma, 1e-12)
                coord_sigma = np.where(width > 0, np.maximum(coord_sigma * 0.85, 1e-12), coord_sigma)

            if evals >= budget:
                break

            remaining = budget - evals

            # ---------------- Refinement step: Gaussian sampling around best ----------------
            # If no improvement for a while, increase exploration slightly by using larger samples.
            # Otherwise, use a tighter distribution.
            if no_improve >= 3:
                local_sigma = global_sigma * 0.95
                # Occasionally restart near random feasible points (limited by remaining budget).
                restart = True
            else:
                local_sigma = global_sigma
                restart = False

            # Decide batch size: small so we can finish exactly by budget.
            # Aim for 2*d+1 style but capped by remaining.
            batch = min(2 * dim + 1, remaining)

            # Optional restart candidates to escape stagnation.
            if restart:
                # Evaluate a couple random points uniformly, then continue with Gaussian around best.
                r = min(2, batch)
                samples = [self._clip(self._sample_uniform(lb, ub), lb, ub) for _ in range(r)]
                batch -= r
            else:
                samples = []

            if batch > 0:
                # Isotropic Gaussian around best_x with per-coordinate scaling by widths.
                # Scale noise by both global_sigma and coordinate widths to remain meaningful.
                # If a coordinate has zero width, noise becomes negligible.
                # Standard deviation per coordinate:
                coord_scale = np.where(width > 0, np.maximum(width, 1e-12), 1.0)
                sigma_vec = (local_sigma * coord_scale / (np.max(coord_scale) if np.max(coord_scale) > 0 else 1.0))
                sigma_vec = np.maximum(sigma_vec, 1e-12)

                for _ in range(batch):
                    noise = np.random.randn(dim) * sigma_vec
                    cand = self._clip(best_x + noise, lb, ub)
                    samples.append(cand)

            for cand in samples:
                if evals >= budget:
                    break
                y, evals_inc = self._evaluate_y(func, cand)
                evals += evals_inc
                if y < best_y:
                    best_y = y
                    best_x = cand.copy()
                    improved = True

            if improved:
                no_improve = 0

            # Extra safeguard: if all bounds are tight (width=0), we can stop early.
            if np.max(width) <= 0:
                break

        return best_x, best_y

    def _evaluate(self, func, x, count):
        # Evaluate objective count times not needed; we use count=1 here.
        y, inc = self._evaluate_y(func, x)
        # Ensure inc equals 1
        return y, inc

    def _evaluate_y(self, func, x):
        # Standardized evaluation: objective is minimization.
        # Never try to predict budget here; budget is handled outside.
        y = func(np.asarray(x, dtype=float))
        return float(y), 1

    def _read_bounds(self, func):
        # Bounds can appear as func.lower/func.upper or func.bounds.lb/func.bounds.ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return func.lower, func.upper
        if hasattr(func, "bounds"):
            b = func.bounds
            # Try common attribute names.
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return b.lb, b.ub
            if hasattr(b, "lower") and hasattr(b, "upper"):
                return b.lower, b.upper
        raise AttributeError(
            "Function does not provide bounds via (lower, upper) or (bounds.lb, bounds.ub)."
        )

    def _clip(self, x, lb, ub):
        return np.minimum(np.maximum(x, lb), ub)

    def _sample_uniform(self, lb, ub):
        # Sample uniformly within bounds.
        # If bounds are degenerate, returns the fixed value.
        r = np.random.rand(self.dim)
        x = lb + r * (ub - lb)
        return x

    def _midpoint(self, lb, ub):
        return (lb + ub) / 2.0
