# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# (derivative-free) that mixes global random exploration with local coordinate
# descent using a small set of candidate points per iteration. It maintains a
# running best solution and progressively shrinks the search radius around the
# best point, while occasionally re-randomizing to escape stagnation.
# Search state: The algorithm tracks the current best point x_best and its
# objective value y_best, the remaining evaluation budget, and an adaptive
# step-size (sigma) controlling how far new candidates are perturbed.
# Candidate generation: In each iteration, it evaluates a small batch of points
# consisting of: (1) random perturbations around x_best, (2) a few
# direction-aligned probes along random coordinate axes, and (3) optional
# pure random points to maintain diversity when progress stalls.
# Selection and replacement: After evaluating the batch, the algorithm selects
# the best-performing candidate and updates x_best/y_best accordingly.
# Adaptation: The step-size sigma is increased slightly when improvement is
# found and decreased when no improvement occurs, bounded within reasonable
# limits derived from the variable ranges.
# Exploration mechanisms: Random perturbations and occasional full random restarts
# help cover the space and reduce the chance of getting trapped early.
# Exploitation mechanisms: Coordinate-axis probes and shrinking sigma promote
# local refinement around the current best.
# Boundary handling: Candidate points are clipped to the feasible bounds
# (box constraints) derived from func.lower/upper or func.bounds.lb/ub.
# Budget strategy: The algorithm strictly decreases a shared evaluation counter
# for each objective call; it never calls the objective more than the provided
# budget. Batches are truncated near the end if needed.
# Closest known influences: The design loosely resembles a simplified CMA-ES-like
# pattern search (adaptive step-size with elitist selection) combined with
# coordinate descent ideas, but implemented without external dependencies.
# Novelty or unusual aspects: The algorithm uses a dynamically sized evaluation
# batch per iteration based on remaining budget and a simple stagnation detector
# that triggers diversity injections.
# Failure modes: If the budget is extremely small, the method behaves mostly
# like random search; on highly non-smooth objectives or deceptive landscapes,
# it may stagnate, though the diversity mechanism attempts to mitigate this.
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
            # No evaluations allowed: return a bounded point deterministically.
            lb, ub = self._read_bounds(func, dim)
            x = (lb + ub) / 2.0
            return x, float("inf")

        lb, ub = self._read_bounds(func, dim)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        # Ensure proper shapes and handle degenerate bounds.
        if lb.shape != (dim,) or ub.shape != (dim,):
            raise ValueError("Bounds must broadcast to shape (dim,).")
        span = np.maximum(ub - lb, 0.0)
        span_nonzero = np.where(span > 0, span, 1.0)

        rng = np.random  # harness sets global seed before each run

        evals_used = 0

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        def eval_obj(x):
            nonlocal evals_used
            if evals_used >= budget:
                # Should never happen due to careful budget accounting.
                return float("inf")
            y = func(x)
            evals_used += 1
            return float(y)

        # Initialize best with a random sample.
        x0 = lb + rng.rand(dim) * (ub - lb)
        y0 = eval_obj(x0)
        x_best = x0
        y_best = y0

        # Adaptive step size: start with a fraction of the box span.
        # If some dimensions have zero span, use 1.0 there to avoid sigma=0.
        sigma = 0.25 * span_nonzero
        sigma = np.maximum(sigma, 1e-12)

        # Iteration parameters (kept compact but robust across dimensions).
        # Batch size is capped and scaled with dim while respecting budget.
        # About 1 + 2*dim probes can be large; cap it.
        max_batch = 16 if dim <= 10 else 32
        min_batch = 4
        # Coordinate probes count: small relative to dim for speed.
        coord_probes = min(max(2, dim // 2), 10)

        no_improve_streak = 0
        # Stagnation threshold relative to budget.
        stagnation_limit = max(6, budget // 10)

        # Continue until evaluations are exhausted.
        while evals_used < budget:
            remaining = budget - evals_used

            # Determine batch size for this round.
            # Start with a small batch; increase slightly if budget allows.
            batch = min(max_batch, remaining)
            batch = max(batch, min_batch) if remaining >= min_batch else remaining
            if batch <= 0:
                break

            candidates = []
            # Always include x_best itself as a reference.
            candidates.append(np.array(x_best, copy=True))

            # 1) Random perturbations around x_best.
            # Use Gaussian steps scaled by sigma and box span.
            # We'll fill until batch size is reached.
            while len(candidates) < batch:
                step = rng.randn(dim) * sigma
                x = clip(x_best + step)
                candidates.append(x)
                if len(candidates) >= batch:
                    break

            # 2) Add a few coordinate-axis probes (replace some random candidates if needed).
            # This strengthens exploitation with limited cost.
            # We'll create up to coord_probes and evaluate them if within batch.
            if batch >= 2 and dim > 0:
                # Prepare a small list of axis directions (random coordinates).
                axes = rng.randint(0, dim, size=coord_probes) if dim > 0 else np.array([], dtype=int)
                # Determine step length along the axis (use fraction of sigma).
                for ax in axes:
                    if len(candidates) >= batch:
                        break
                    # Try +/- along selected axis.
                    step_len = sigma[ax] * (0.5 + 0.5 * rng.rand())
                    # Alternate signs deterministically by sequence to reduce randomness.
                    sign = -1.0 if (len(candidates) % 2 == 0) else 1.0
                    x = np.array(x_best, copy=True)
                    x[ax] = x_best[ax] + sign * step_len
                    x = clip(x)
                    candidates.append(x)

            # 3) If stagnated, inject diversity by including some pure random points.
            if no_improve_streak >= stagnation_limit and remaining >= 2:
                inject_n = min(3, batch)  # small number to keep evaluations bounded
                # Replace the last few candidates with random points.
                for i in range(1, inject_n + 1):
                    if len(candidates) - i < 0:
                        break
                    candidates[-i] = lb + rng.rand(dim) * (ub - lb)
                # Reset streak after injection attempt (progress may or may not happen).
                no_improve_streak = max(0, no_improve_streak - stagnation_limit // 2)

            # Evaluate all candidates (truncate if somehow over budget).
            best_local_x = None
            best_local_y = float("inf")
            for x in candidates:
                if evals_used >= budget:
                    break
                y = eval_obj(x)
                if y < best_local_y:
                    best_local_y = y
                    best_local_x = x

            # Update global best.
            if best_local_x is not None and best_local_y < y_best:
                x_best = best_local_x
                y_best = best_local_y
                no_improve_streak = 0
                # Increase sigma slightly to explore around the improved region.
                # Keep it bounded by a fraction of total span.
                max_sigma = np.where(span_nonzero > 0, 0.5 * span_nonzero, 1.0)
                sigma = np.minimum(sigma * 1.15, max_sigma)
            else:
                no_improve_streak += 1
                # Decrease sigma to exploit locally.
                sigma = sigma * 0.85
                # Prevent sigma from collapsing too far (still allow movement).
                min_sigma = 1e-6 * span_nonzero
                sigma = np.maximum(sigma, min_sigma)

            # If sigma is very small and we haven't improved, force some diversity
            # by re-centering around a random point (without increasing evals).
            # The next loop will include random injections if stagnation persists.
            if no_improve_streak >= stagnation_limit and evals_used < budget:
                # Move best center halfway towards a random point to avoid
                # pathological clipping traps.
                xr = lb + rng.rand(dim) * (ub - lb)
                x_best = clip(0.5 * x_best + 0.5 * xr)

        return x_best, y_best

    def _read_bounds(self, func, dim):
        # Priority: func.lower/func.upper, else func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.array(func.lower, dtype=float)
            ub = np.array(func.upper, dtype=float)
            return self._broadcast_bounds(lb, ub, dim)

        if hasattr(func, "bounds"):
            b = func.bounds
            lb = np.array(getattr(b, "lb"), dtype=float)
            ub = np.array(getattr(b, "ub"), dtype=float)
            return self._broadcast_bounds(lb, ub, dim)

        raise AttributeError(
            "Objective function must provide bounds via "
            "func.lower/func.upper or func.bounds.lb/func.bounds.ub."
        )

    @staticmethod
    def _broadcast_bounds(lb, ub, dim):
        # Accept scalar, (dim,), or broadcastable arrays.
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        if lb.shape == () and ub.shape == ():
            return np.full(dim, float(lb)), np.full(dim, float(ub))
        # Broadcast to (dim,)
        try:
            lb_b = np.broadcast_to(lb, (dim,)).astype(float, copy=False)
            ub_b = np.broadcast_to(ub, (dim,)).astype(float, copy=False)
        except Exception as e:
            raise ValueError("Bounds are not broadcastable to shape (dim,).") from e

        # Make sure lb <= ub where possible; swap if inverted.
        lb2 = np.minimum(lb_b, ub_b)
        ub2 = np.maximum(lb_b, ub_b)
        return lb2, ub2
