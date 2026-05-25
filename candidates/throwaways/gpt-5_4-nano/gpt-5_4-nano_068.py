# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# using a mix of random search and local refinement. It maintains a current
# best solution and a Gaussian sampling distribution around it.
# Search state: The algorithm tracks the best point (best_x), its objective
# value (best_y), and a step size (sigma) controlling exploration radius.
# Candidate generation: Each iteration generates a batch of candidate points
# by sampling from a normal distribution centered at the current best, plus
# occasional purely random points to encourage global exploration. A simple
# coordinate-wise jitter is also used early on to improve robustness.
# Selection and replacement: From all candidates generated in an iteration,
# the algorithm chooses the best (lowest objective value) to update best_x and
# best_y. The sampling distribution is then centered on the updated best.
# Adaptation: sigma decays when improvement is found and grows slightly when
# no improvement occurs, balancing exploration and exploitation.
# Exploration mechanisms: Early random probing and periodic uniform random
# candidates mitigate premature convergence and help escape local minima.
# Exploitation mechanisms: Most candidates are sampled around best_x using the
# adaptive Gaussian distribution for local search.
# Boundary handling: After sampling, points are clipped to the provided bounds
# (lower/upper). If the bounds are degenerate in a dimension, that component
# is kept fixed.
# Budget strategy: The algorithm uses the provided evaluation budget exactly.
# It precomputes an iteration count and batch size so the total number of
# objective evaluations never exceeds budget (and always equals budget when
# possible, otherwise less if budget is very small).
# Closest known influences: The design is reminiscent of CMA-ES-lite / evolution
# strategies in spirit (best-centered Gaussian sampling with step-size
# adaptation), simplified to be robust, dependency-free, and compact.
# Novelty or unusual aspects: It includes an evaluation-aware batching scheme
# that strictly respects the budget, plus a fallback coordinate jitter near the
# start for better behavior in small dimensions.
# Failure modes: In very high dimensions with tight budgets, the algorithm may
# struggle due to limited exploration. If the objective is extremely noisy or
# highly non-smooth, the sigma adaptation may oscillate and convergence could be
# slow.
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

        # --- Read bounds from func ---
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub")

        if lb.shape != (dim,) or ub.shape != (dim,):
            lb = np.reshape(lb, (dim,))
            ub = np.reshape(ub, (dim,))

        # Degenerate bounds handling
        span = ub - lb
        span = np.where(np.isfinite(span), span, 0.0)
        # For zero span dimensions, keep them fixed at lb/ub.
        fixed = span <= 0

        rng = np.random

        def clip_to_bounds(x):
            # Clip all dims; for fixed dims this keeps them constant anyway.
            return np.minimum(np.maximum(x, lb), ub)

        def sample_uniform():
            # Uniform sampling within bounds; fixed dims collapse to constant.
            # Use np.random.random for standard-library restriction.
            u = rng.random(dim)
            x = lb + u * span
            if np.any(fixed):
                x = np.where(fixed, lb, x)
            return x

        # If budget is extremely small, do the minimum sensible evaluations.
        evals = 0
        best_x = sample_uniform()
        best_y = float(func(best_x))
        evals += 1

        if budget <= 1:
            return best_x, best_y

        # Initial sigma: proportional to typical range.
        # Use a robust scale so it works across dimensions and ranges.
        # If span is zero in all dims, sigma becomes 0 and search is trivial.
        nonzero_span = span[~fixed]
        if nonzero_span.size == 0:
            # All variables are fixed; objective is constant w.r.t. x.
            # We already evaluated once.
            return best_x, best_y

        typical = np.median(np.abs(nonzero_span))
        sigma = 0.3 * typical
        sigma = float(max(sigma, 1e-12))

        # Strict evaluation-aware batching
        # Choose a batch size that is small enough to stay responsive but efficient.
        # It must respect the budget exactly.
        remaining = budget - evals
        base_batch = 8
        batch_size = min(base_batch, max(1, remaining))
        # Number of iterations (batches) we can afford
        # We'll dynamically adjust the last batch size.
        iters = int(np.ceil(remaining / batch_size))

        # Factors for adaptation
        improve_shrink = 0.82
        no_improve_grow = 1.08

        # Candidate counts: we generate "k_local" around best plus "k_random"
        # occasional global points.
        for it in range(iters):
            if evals >= budget:
                break

            remaining = budget - evals
            k = min(batch_size, remaining)

            # Decide split for exploration/exploitation.
            # Early iterations explore more; later exploit more.
            progress = it / max(1, iters - 1)
            # k_random decreases with progress
            k_random = int(round(k * (0.35 * (1.0 - progress))))
            k_random = min(k_random, k)
            k_local = k - k_random

            candidates = []

            # --- Exploration candidates: uniform random ---
            for _ in range(k_random):
                candidates.append(sample_uniform())

            # --- Exploitation candidates: Gaussian around best ---
            # Use diagonal-ish isotropic sampling with adaptive sigma.
            for j in range(k_local):
                z = rng.standard_normal(dim)
                # Optional early coordinate jitter for robustness
                if it == 0 and j < min(2, k_local):
                    # Coordinate-wise jitter reduces risk of missing narrow directions.
                    # It effectively mixes an axis-aligned perturbation with isotropic noise.
                    axis = rng.randint(0, dim)
                    z[axis] += rng.standard_normal() * 2.0
                x = best_x + sigma * z
                if np.any(fixed):
                    x = np.where(fixed, lb, x)
                x = clip_to_bounds(x)
                candidates.append(x)

            # Evaluate and select best candidate
            current_best_y = best_y
            improved = False
            local_best_x = best_x
            local_best_y = best_y

            for x in candidates:
                if evals >= budget:
                    break
                y = float(func(x))
                evals += 1
                if y < local_best_y:
                    local_best_y = y
                    local_best_x = x
                    improved = True

            if local_best_y < best_y:
                best_y = local_best_y
                best_x = local_best_x

            # Adapt sigma based on improvement and iteration stage.
            # Even if no improvement, allow gradual expansion to re-explore.
            if improved:
                sigma *= improve_shrink
            else:
                sigma *= no_improve_grow

            # Keep sigma within reasonable bounds relative to span.
            # Upper bound avoids wasting evaluations far outside the space.
            # Lower bound avoids stalling numerically.
            max_span = np.max(nonzero_span)
            sigma_max = 0.5 * max_span if max_span > 0 else sigma
            sigma_min = 1e-12
            sigma = float(min(max(sigma, sigma_min), sigma_max))

            # If sigma is extremely small and we're close to a fixed point, we can
            # still continue because there might be exact bound corners.
            # (No extra early stopping: must respect budget policy.)

        return best_x, best_y
