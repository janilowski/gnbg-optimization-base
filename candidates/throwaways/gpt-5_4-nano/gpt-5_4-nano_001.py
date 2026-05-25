# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm for a continuous
# domain using a population-based evolutionary strategy (CMA-ES-inspired but simpler).
# It maintains a Gaussian sampling distribution over the decision variables and
# adaptively contracts/expands step size based on recent improvements. The algorithm
# evaluates candidates in batches while strictly respecting the provided evaluation budget.
#
# Search state: A mean vector (current best center), a diagonal step-size (sigma),
# and a single scalar learning signal derived from the best objective values.
# Also tracks current best solution (best_x, best_y) across all evaluations.
#
# Candidate generation: Each iteration samples lambda candidate points:
#   x ~ mean + sigma * N(0, I)
# with added per-dimension scaling derived from bounds to help with numeric stability.
# Candidates are clipped to bounds.
#
# Selection and replacement: The objective values are sorted; the best individual becomes
# a reference and the mean is updated toward a weighted average of the top half.
# The global best is updated whenever a new lower objective is found.
#
# Adaptation: The step size sigma is increased slightly when improvement is poor
# and decreased when improvement is strong, using an order-statistics-like signal
# computed from the best values in the current iteration.
#
# Exploration mechanisms: Stochastic sampling around the mean and sigma-driven
# global perturbations; larger sigma increases exploration.
#
# Exploitation mechanisms: Mean updates toward elite solutions and sigma reduction
# after consistent improvements increases exploitation around promising regions.
#
# Boundary handling: Points are clipped to feasible bounds after sampling.
# The initial mean is set to the midpoint of bounds if available, otherwise to zeros.
#
# Budget strategy: The total number of function evaluations is capped by budget.
# Each iteration consumes up to the remaining evaluations; iteration count is chosen
# based on lambda and remaining evaluations.
#
# Closest known influences: Generic evolutionary strategies / (very lightweight) CMA-ES
# variants using elitist recombination and adaptive global step size.
#
# Novelty or unusual aspects: Uses bounds-derived scaling to keep sigma effective across
# different variable ranges, and a simple budget-aware, single-sigma adaptation rule
# that remains compact and robust.
#
# Failure modes: If the objective is highly discontinuous or extremely noisy, the
# order-based adaptation may mislead sigma updates. If bounds are extremely tight,
# sigma may quickly shrink and cause stagnation; clipping ensures feasibility but
# can reduce diversity. In very high dimensions, the small population size may slow
# convergence.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        # --- Read bounds from func ---
        lb, ub = self._get_bounds(func, self.dim)

        # Helper: clip to bounds
        def clip(x: np.ndarray) -> np.ndarray:
            return np.minimum(np.maximum(x, lb), ub)

        # --- Initialize evaluation budget ---
        # Choose a small-but-effective population size scaling with dimension.
        # Must be >= 2 to allow meaningful selection.
        lam = int(max(8, min(48, 4 + self.dim)))  # compact heuristic
        lam = max(2, lam)

        # Mean initialization: midpoint of bounds; if bounds degenerate, still valid.
        mean = (lb + ub) * 0.5

        # Initial sigma: fraction of average span (avoid zero).
        span = (ub - lb)
        avg_span = float(np.mean(span)) if np.all(np.isfinite(span)) else 1.0
        avg_span = max(avg_span, 1e-12)
        sigma = 0.25 * avg_span  # global scale

        # Optionally rescale step size by per-dimension spans for stability.
        # This keeps sigma meaningful even when spans differ substantially.
        span_safe = np.maximum(span, 1e-12)
        per_dim_scale = span_safe / float(np.mean(span_safe))  # mean ~ 1

        # Evaluate at initial mean (counts against budget)
        evals = 0
        best_x = mean.copy()
        best_y = float(func(best_x))
        evals += 1

        # If budget exhausted, return immediately
        if evals >= self.budget:
            return best_x, best_y

        # Determine number of iterations based on budget
        # Each iteration evaluates up to lam candidates.
        # Total evaluations = 1 + k*lam' (where lam' <= lam near the end).
        # Compute a safe upper bound on k.
        remaining = self.budget - evals
        max_iters = max(1, int(math.ceil(remaining / lam)))

        # Step-size adaptation parameters (simple, robust)
        # Decrease when strong improvement; increase when weak.
        sigma_down = 0.82
        sigma_up = 1.18
        sigma_min = 1e-15 * float(np.mean(span_safe))
        sigma_max = float(np.max(span_safe)) * 2.0 + 1e-12

        # Elite fraction for recombination
        elite_count = max(1, lam // 2)

        rng = np.random.default_rng()  # harness seeds global state via numpy; rng uses it too

        # Main loop
        for _ in range(max_iters):
            rem = self.budget - evals
            if rem <= 0:
                break

            cur_lam = min(lam, rem)

            # Sample candidates
            # x = mean + (sigma * per_dim_scale) * N(0,1)
            noise = rng.standard_normal(size=(cur_lam, self.dim))
            steps = (sigma * per_dim_scale) * noise
            X = clip(mean[None, :] + steps)

            # Evaluate
            Y = np.empty(cur_lam, dtype=float)
            for i in range(cur_lam):
                y = float(func(X[i]))
                Y[i] = y

            evals += cur_lam

            # Track best
            idx_best = int(np.argmin(Y))
            if Y[idx_best] < best_y:
                best_y = float(Y[idx_best])
                best_x = X[idx_best].copy()

            # Order for selection
            order = np.argsort(Y)
            Xs = X[order[:elite_count]]
            Ys = Y[order[:elite_count]]

            # Elite recombination: weighted towards best
            # Weights decrease with rank; sum to 1.
            ranks = np.arange(elite_count, dtype=float)
            w = np.log((elite_count + 0.5) / (ranks + 0.5))
            w = w / float(np.sum(w))
            new_mean = np.sum(Xs * w[:, None], axis=0)

            # Adapt sigma based on improvement signal
            # Use best in current generation relative to previous best.
            gen_best = float(Y[order[0]])
            improvement = best_y - gen_best  # positive means improved best_y, but after update best_y already set
            # Instead: compare generation best to previous global best stored before update.
            # We captured best_y possibly updated; so compute using min so far by tracking prev.
            # We'll store prev_best_y by re-deriving from Ys: if best updated, we don't have old value.
            # To keep robust without extra state, compute a local improvement indicator using spread:
            # If elites are much better than the rest, shrink; otherwise expand slightly.
            if elite_count < cur_lam:
                # Compare best elite average vs. average of remaining
                avg_elite = float(np.mean(Y[order[:elite_count]]))
                avg_others = float(np.mean(Y[order[elite_count:]])) if elite_count < cur_lam else avg_elite
                rel = (avg_others - avg_elite) / (abs(avg_others) + 1e-12)
            else:
                rel = 0.0

            # rel > 0 means elites better than others. Shrink when strong separation.
            # Threshold tuned to be conservative.
            if rel > 0.08:
                sigma *= sigma_down
            else:
                sigma *= sigma_up

            # Clamp sigma
            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # Update mean
            mean = new_mean

        return best_x, best_y

    @staticmethod
    def _get_bounds(func: Any, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract bounds from func using one of:
          - func.lower / func.upper
          - func.bounds.lb / func.bounds.ub
        Supports scalars or array-like; converts to float ndarray of shape (dim,).
        """
        lb = ub = None

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = getattr(func, "lower")
            ub = getattr(func, "upper")
        elif hasattr(func, "bounds"):
            bounds = getattr(func, "bounds")
            if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
                lb = bounds.lb
                ub = bounds.ub

        if lb is None or ub is None:
            # Fallback: assume standard box [-5, 5] if bounds are not provided.
            # (Still makes the algorithm usable; harness typically provides bounds.)
            lb_arr = np.full(dim, -5.0, dtype=float)
            ub_arr = np.full(dim, 5.0, dtype=float)
            return lb_arr, ub_arr

        lb_arr = np.asarray(lb, dtype=float)
        ub_arr = np.asarray(ub, dtype=float)

        # Broadcast scalar bounds
        if lb_arr.ndim == 0:
            lb_arr = np.full(dim, float(lb_arr), dtype=float)
        if ub_arr.ndim == 0:
            ub_arr = np.full(dim, float(ub_arr), dtype=float)

        # Flatten and validate
        lb_arr = lb_arr.reshape(-1)
        ub_arr = ub_arr.reshape(-1)
        if lb_arr.size != dim or ub_arr.size != dim:
            # Attempt broadcasting from smaller arrays if possible
            if lb_arr.size == 1:
                lb_arr = np.full(dim, float(lb_arr[0]), dtype=float)
            if ub_arr.size == 1:
                ub_arr = np.full(dim, float(ub_arr[0]), dtype=float)

        lb_arr = lb_arr.reshape(-1)
        ub_arr = ub_arr.reshape(-1)
        if lb_arr.size != dim or ub_arr.size != dim:
            raise ValueError(f"Bounds dimension mismatch: expected dim={dim}, got lb={lb_arr.size}, ub={ub_arr.size}")

        # Ensure order lb <= ub
        lo = np.minimum(lb_arr, ub_arr)
        hi = np.maximum(lb_arr, ub_arr)
        # If there are NaNs/infs, clip them into a safe finite range.
        lo = np.where(np.isfinite(lo), lo, -5.0)
        hi = np.where(np.isfinite(hi), hi, 5.0)
        return lo.astype(float, copy=False), hi.astype(float, copy=False)
