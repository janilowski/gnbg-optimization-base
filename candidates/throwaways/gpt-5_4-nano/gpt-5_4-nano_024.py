# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a simple yet robust black-box minimization algorithm using
# iterative Gaussian sampling around the current best point, with step-size
# adaptation and occasional restart. It is designed to work in any dimension
# and to respect a strict evaluation budget.
# Search state: Tracks the current best solution x_best and its objective value
# y_best, along with the remaining evaluation budget and an adaptive step-size sigma.
# Candidate generation: Each iteration samples a small batch of candidate points
# from a normal distribution centered at x_best with scale sigma. Candidates are
# clipped to the valid bounds.
# Selection and replacement: Evaluates all candidates and greedily updates x_best
# to the best candidate in the batch. If improvement occurs, sigma is slightly
# reduced; otherwise sigma is increased to encourage broader exploration.
# Adaptation: Uses a success-based step-size adaptation similar in spirit to
# 1/5th rule: success triggers shrinkage, failure triggers expansion. Also includes
# a rare restart when progress stalls.
# Exploration mechanisms: When sigma is large or after restarts, sampling explores
# more widely. Randomness comes from NumPy’s RNG (the harness controls seeding).
# Exploitation mechanisms: As improvements are found, sigma shrinks, focusing
# sampling near promising areas.
# Boundary handling: All sampled points are clipped to the provided bounds.
# Budget strategy: Uses a budget-aware loop. The total number of function evaluations
# never exceeds the provided budget. Each iteration evaluates at most the remaining
# evaluations.
# Closest known influences: Inspired by evolution strategies (ES) / CMA-like
# but simplified to a single-point Gaussian resampling with adaptive sigma.
# Novelty or unusual aspects: Includes a deterministic budget-safe batching strategy,
# dimension-scaled sigma initialization, and a stall-triggered restart to reduce
# the chance of getting stuck in flat regions.
# Failure modes: If the objective is extremely noisy, the greedy update and sigma
# adaptation may oscillate or converge slowly. In highly constrained or very narrow
# feasible regions, clipping may reduce effective search diversity.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim

        lb, ub = self._read_bounds(func, dim)
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        # Budget safety: ensure we never exceed the given evaluations.
        # We will count evaluations precisely.
        max_evals = max(1, int(self.budget))
        evals = 0

        rng = np.random

        # Initialize sigma using the scale of the domain.
        # Use a conservative fraction of the range; avoid zeros.
        domain = np.maximum(ub - lb, 1e-12)
        sigma0 = 0.3 * float(np.mean(domain))
        if not np.isfinite(sigma0) or sigma0 <= 0:
            sigma0 = 1.0

        # Start from a random point within bounds.
        x_best = lb + rng.random(dim) * (ub - lb)
        y_best = func(x_best)
        evals += 1

        # If budget is exhausted, return immediately.
        if evals >= max_evals:
            return x_best, float(y_best)

        # Batch size: small, dimension-aware, budget-aware.
        # Larger batches can speed progress on smooth functions.
        # Must remain >=1.
        base_batch = max(2, min(8, 1 + dim // 2))
        batch = int(base_batch)

        sigma = sigma0
        stall_iters = 0
        best_y_prev = y_best

        # Loop until budget is exhausted.
        # Each loop evaluates up to `batch` candidates or remaining budget.
        while evals < max_evals:
            remaining = max_evals - evals
            k = min(batch, remaining)

            # Generate candidates around x_best.
            # Use isotropic Gaussian with dimension-scaled steps:
            # sigma is global; direction noise is iid per dimension.
            # Use float64 for numerical stability.
            noise = rng.standard_normal((k, dim))
            X = x_best[None, :] + sigma * noise

            # Boundary handling: clip to feasible box.
            X = np.minimum(np.maximum(X, lb[None, :]), ub[None, :])

            # Evaluate candidates (counted).
            ys = np.empty(k, dtype=float)
            for i in range(k):
                ys[i] = func(X[i])
            evals += k

            # Greedy selection: pick best candidate.
            idx = int(np.argmin(ys))
            y_cand = float(ys[idx])
            x_cand = X[idx]

            improved = y_cand < y_best

            # Update best.
            if improved:
                x_best = x_cand
                y_best = y_cand
                stall_iters = 0
            else:
                stall_iters += 1

            # Success-based sigma adaptation.
            # Shrink when improving, expand otherwise.
            if improved:
                sigma *= 0.82
            else:
                sigma *= 1.18

            # Keep sigma within reasonable bounds relative to the domain.
            # Avoid too tiny sigma (numerical stagnation) or too huge (mostly clipping).
            sigma_min = 1e-12 * float(np.mean(domain))
            sigma_max = 1.0 * float(np.max(domain))
            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # Rare restart: if no improvement for a while, re-center randomly and
            # increase sigma to re-explore. The threshold depends on dimension.
            if stall_iters >= max(4, 2 + dim // 2):
                # Ensure we do not exceed budget: restart is free (no evaluations).
                x_best = lb + rng.random(dim) * (ub - lb)
                y_best = func(x_best)
                evals += 1
                # If this evaluation exhausts the budget, stop.
                if evals >= max_evals:
                    return x_best, float(y_best)
                # Reset sigma upward to explore around new center.
                sigma = sigma0
                stall_iters = 0
                # Track previous best (not strictly needed, but helps clarity).
                best_y_prev = y_best

            # Also allow an early exit if progress is numerically negligible.
            # (Does not change eval count behavior.)
            if y_best >= best_y_prev - 1e-15:
                best_y_prev = y_best
            else:
                best_y_prev = y_best

        return x_best, float(y_best)

    @staticmethod
    def _read_bounds(func, dim):
        # Support several possible formats:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        # Bounds may be scalars or array-like. Convert to numpy arrays.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(getattr(func, "lower"), dtype=float)
            ub = np.asarray(getattr(func, "upper"), dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError(
                "Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )

        # Broadcast/reshape to (dim,)
        if lb.shape == () and ub.shape == ():
            lb = np.full(dim, float(lb))
            ub = np.full(dim, float(ub))
        else:
            lb = np.asarray(lb, dtype=float).reshape(-1)
            ub = np.asarray(ub, dtype=float).reshape(-1)
            if lb.size != dim or ub.size != dim:
                # Try broadcasting from length-1 vectors.
                if lb.size == 1:
                    lb = np.full(dim, float(lb[0]))
                if ub.size == 1:
                    ub = np.full(dim, float(ub[0]))
                if lb.size != dim or ub.size != dim:
                    raise ValueError(
                        f"Bounds have incompatible shapes. Expected length {dim}, got lb={lb.size}, ub={ub.size}."
                    )

        # Ensure lb <= ub
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        return lb2, ub2
