# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# (Algorithm class) intended for GNBG-style benchmark harnesses. It performs
# a budget-aware derivative-free search using a mixture of simplex-like
# direction probes and stochastic coordinate perturbations. The objective is
# minimization.
#
# Search state: The algorithm maintains a current best point x_best with its
# function value y_best, plus a shrinking step size (sigma) and a small set of
# candidate points evaluated around the current best in each iteration.
#
# Candidate generation: Each iteration generates candidates by (1) sampling
# random directions and evaluating along positive/negative steps, and (2)
# performing a short randomized coordinate-wise perturbation. Candidate
# points are created by adding scaled perturbations to the current best.
#
# Selection and replacement: After each candidate evaluation, if the new value
# is smaller, it becomes the new best point. At the end of each iteration,
# the algorithm updates the step size based on whether it made progress.
#
# Adaptation: The step size sigma is reduced when progress is not observed
# (multiplicative decay), and increased modestly when improvements occur to
# help escape shallow local minima.
#
# Exploration mechanisms: Random direction probing (including both signs) and
# coordinate perturbations encourage exploring the space.
#
# Exploitation mechanisms: All candidate generation is anchored to the current
# best point, biasing the search towards regions that already show good fitness.
#
# Boundary handling: Candidate points are clipped to the provided bounds
# before evaluation, ensuring feasibility even when proposals go outside.
#
# Budget strategy: The number of iterations and the number of evaluations per
# iteration are computed to never exceed the provided evaluation budget. The
# algorithm also short-circuits if the budget is exhausted.
#
# Closest known influences: The design loosely resembles evolution strategies /
# coordinate search hybrids and a bracketed random directional search, but is
# implemented compactly without external dependencies.
#
# Novelty or unusual aspects: The implementation dynamically allocates evaluation
# effort across iterations using the remaining budget and uses a progress-
# dependent sigma schedule rather than a fixed schedule.
#
# Failure modes: If the objective is extremely noisy or has flat regions,
# progress may be infrequent, leading to sigma shrinking too much; the modest
# sigma increase on improvement helps mitigate this. If bounds are very tight,
# clipping may reduce effective search diversity.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        lb, ub = self._read_bounds(func, self.dim)
        d = self.dim

        # Handle degenerate bounds gracefully
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            raise ValueError("Bounds must be finite.")
        if np.any(ub < lb):
            raise ValueError("Upper bounds must be >= lower bounds.")

        evals_used = 0

        def eval_at(x):
            nonlocal evals_used
            if evals_used >= self.budget:
                # Never exceed budget; if called, return +inf to discourage.
                return np.inf
            y = func(x)
            evals_used += 1
            return float(y)

        # Initial point: random within bounds (harness seeds numpy for determinism)
        if self.budget <= 0:
            # No evaluations possible; return a feasible point and +inf
            x0 = lb.copy()
            x0 = np.where(lb == ub, lb, lb + 0.0 * (ub - lb))
            return x0, np.inf

        x_best = lb + np.random.rand(d) * (ub - lb) if np.any(ub > lb) else lb.copy()
        y_best = eval_at(np.array(x_best, dtype=float))

        # Step size initialized relative to domain scale.
        # Use a robust scale estimate to work across dimensions.
        domain = ub - lb
        dom_scale = float(np.max(domain)) if np.any(domain > 0) else 1.0
        sigma = 0.25 * dom_scale if dom_scale > 0 else 0.25

        # Iteration planning: budget-aware, with at least one evaluation (already done).
        # Remaining evaluations:
        remaining = self.budget - evals_used
        if remaining <= 0:
            return x_best, y_best

        # Decide how many candidates to try per iteration (small, robust).
        # We use at most 2*d + 2 evaluations per iteration: one baseline is already done.
        # Keep it compact and budget-aware.
        max_per_iter = int(min(max(2 * d + 2, 6), remaining))
        # Use a conservative number of iterations; more iterations generally help adapt sigma.
        iters = max(1, remaining // max_per_iter)
        # Ensure total doesn't exceed budget: we'll compute per-iteration caps dynamically.

        # Multipliers controlling adaptation
        decay = 0.82
        growth = 1.18

        # Precompute a normalized random direction generator for speed/readability.
        def sample_unit_vectors(k):
            # Returns k unit vectors of shape (k, d)
            v = np.random.randn(k, d)
            norms = np.linalg.norm(v, axis=1)
            # Avoid division by zero: if a vector is ~0, replace with a new one
            zero_mask = norms < 1e-12
            if np.any(zero_mask):
                v[zero_mask] = np.random.randn(np.sum(zero_mask), d)
                norms = np.linalg.norm(v, axis=1)
            return v / norms[:, None]

        # Main loop
        for _ in range(iters):
            if evals_used >= self.budget:
                break

            # Remaining evaluations for this iteration.
            rem = self.budget - evals_used
            # Allocate candidates: directional probes + coordinate perturbations.
            # Keep candidate count <= rem.
            # Directional probes try both signs along random directions.
            # Each direction uses 2 evaluations (plus the direction anchored at best).
            # Coordinate perturbation uses 1 or 2 evaluations depending on rem.
            # We'll budget as follows:
            # Use as many directional pairs as fit.
            # dir_pairs * 2 + coord_evals <= rem
            dir_pairs = max(0, min(d, (rem // 2)))  # each pair uses 2 evals
            coord_evals = 0
            # Ensure we don't overshoot; keep at least some coordinate exploration.
            if rem - 2 * dir_pairs > 0:
                coord_evals = rem - 2 * dir_pairs
                # Cap to a small number to stay robust/compact.
                coord_evals = int(min(coord_evals, 2 + d // 2))

            improved = False

            # Directional probing (both signs)
            if dir_pairs > 0:
                # Generate unit vectors and create proposals scaled by sigma.
                U = sample_unit_vectors(dir_pairs)  # (dir_pairs, d)
                step = sigma * (0.5 + 0.5 * np.random.rand(dir_pairs))  # randomize magnitude in [0.5,1.0]*sigma
                # Two-sided proposals: x_best +/- step*u
                for i in range(dir_pairs):
                    if evals_used >= self.budget:
                        break
                    xi = x_best - step[i] * U[i]
                    yi = eval_at(np.clip(xi, lb, ub))
                    if yi < y_best:
                        x_best, y_best = np.array(xi, dtype=float), yi
                        improved = True

                    if evals_used >= self.budget:
                        break
                    xj = x_best + step[i] * U[i]
                    yj = eval_at(np.clip(xj, lb, ub))
                    if yj < y_best:
                        x_best, y_best = np.array(xj, dtype=float), yj
                        improved = True

            # Coordinate perturbations: randomly select coordinates and perturb by sigma
            # This helps in axis-aligned landscapes or in clipped regions.
            # Use up to coord_evals evaluations.
            if evals_used < self.budget and coord_evals > 0:
                # Choose coordinates with probability proportional to domain size
                # (larger ranges get slightly more exploration).
                if np.all(domain <= 0):
                    coord_probs = np.ones(d) / d
                else:
                    # Add epsilon to avoid zero probs when domain is zero.
                    eps = 1e-12
                    w = np.maximum(domain, eps)
                    coord_probs = w / np.sum(w)

                # For each evaluation, perturb a (possibly repeated) coordinate.
                # Include both signs when possible within budget for symmetry.
                # We'll do one eval per loop, but sometimes use two (if budget allows).
                for _k in range(coord_evals):
                    if evals_used >= self.budget:
                        break
                    j = int(np.random.choice(d, p=coord_probs))
                    # Draw sign and scale
                    sign = 1.0 if np.random.rand() < 0.5 else -1.0
                    scale = (0.5 + np.random.rand()) * sigma
                    x_new = np.array(x_best, copy=True)
                    x_new[j] = x_new[j] + sign * scale
                    x_new = np.clip(x_new, lb, ub)
                    y_new = eval_at(x_new)
                    if y_new < y_best:
                        x_best, y_best = x_new, y_new
                        improved = True

            # Sigma adaptation
            if improved:
                sigma = min(sigma * growth, 0.5 * dom_scale if dom_scale > 0 else sigma * growth)
            else:
                sigma = sigma * decay

            # If sigma becomes extremely small relative to bounds, stop early.
            if sigma <= 1e-15 * (dom_scale if dom_scale > 0 else 1.0):
                break

        return np.array(x_best, dtype=float), float(y_best)

    @staticmethod
    def _read_bounds(func, dim):
        # Priority: func.lower/func.upper, else func.bounds.lb/ub
        lb = ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        if lb is None or ub is None:
            raise ValueError("Bounds not found. Provide func.lower/func.upper or func.bounds.lb/ub.")

        # Broadcast/reshape to dim
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != dim or ub.size != dim:
            # Allow scalars
            if lb.size == 1:
                lb = np.full(dim, float(lb[0]))
            if ub.size == 1:
                ub = np.full(dim, float(ub[0]))
        if lb.size != dim or ub.size != dim:
            raise ValueError(f"Bounds size mismatch: expected dim={dim}, got lb={lb.size}, ub={ub.size}.")
        return lb, ub
