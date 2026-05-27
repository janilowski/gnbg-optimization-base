# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# based on population-based sampling with a shrinking “trust region” and
# occasional random restarts. It works for arbitrary dimensions using only
# numpy.
# Search state: The algorithm maintains a small population of candidate points,
# tracks their objective values, and keeps the current best solution found so far.
# Candidate generation: Each iteration generates new candidates around the best
# point using Gaussian perturbations whose scale is controlled by a trust-region
# radius. Additionally, with a small probability it generates uniformly random
# points across the bounds (restart/exploration).
# Selection and replacement: After evaluating all candidates for the iteration,
# the algorithm selects the best among them to update the global best. It also
# uses the current best to form the next iteration’s sampling center.
# Adaptation: The trust-region radius shrinks when improvement is observed and
# grows slightly when progress stalls, balancing global exploration and local
# exploitation.
# Exploration mechanisms: Periodic/conditional uniform sampling across bounds
# provides exploration and helps escape local minima.
# Exploitation mechanisms: Gaussian sampling concentrated around the best point
# performs local search.
# Boundary handling: All candidate points are clipped to the provided box
# constraints before evaluation.
# Budget strategy: The total number of function evaluations is strictly capped
# by the provided budget. Evaluations are counted explicitly; if an iteration
# would exceed the remaining budget, the algorithm reduces the candidate count
# for that iteration.
# Closest known influences: Combines ideas from evolution strategies (sample and
# select), trust-region radius adaptation, and restart-based exploration.
# Novelty or unusual aspects: The trust-region radius is updated using a simple
# improvement trigger and a stagnation counter, making the behavior robust
# without needing algorithm-specific tuning per dimension.
# Failure modes: If the objective landscape is extremely flat or highly noisy,
# the radius adaptation may stall; however, restarts and continuous random
# exploration mitigate this. In very high dimensions, the fixed population size
# may need more budget to be effective.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

        # Population size: chosen to be small but dimension-aware.
        # Must be >= 1 and not exceed budget.
        pop = 4 + int(np.ceil(np.log2(max(2, self.dim))))
        self.pop_size = max(1, min(pop, self.budget if self.budget > 0 else 1))

        # How often to attempt exploration (restart) within an iteration.
        self.explore_prob = 0.15

        # Stagnation handling for trust-region adaptation.
        self.stagnation_limit = 6

        # Small epsilon to avoid degenerate scales.
        self.eps = 1e-12

    def __call__(self, func):
        # Read bounds from func
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        if lb is None or ub is None:
            raise AttributeError("Objective function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub")

        n = self.dim
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.shape[0] != n or ub.shape[0] != n:
            raise ValueError("Bounds dimensionality does not match dim")

        # Handle pathological bounds where ub < lb
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo
        span = np.where(np.isfinite(span), span, 0.0)
        span = np.maximum(span, 0.0)

        # Evaluation budget management
        max_evals = max(0, int(self.budget))
        evals_used = 0

        def eval_point(x):
            nonlocal evals_used
            if evals_used >= max_evals:
                # Should never be called beyond budget, but keep robust.
                return np.inf
            evals_used += 1
            return float(func(x))

        # Initial sampling: include best of a small random set.
        # Also evaluate the mid-point as a deterministic anchor when feasible.
        mid = lo + 0.5 * (hi - lo)
        # Trust-region radius: start as a fraction of average span.
        avg_span = float(np.mean(span)) if n > 0 else 0.0
        radius = 0.5 * avg_span if avg_span > 0 else 1.0

        best_x = mid.copy()
        best_y = eval_point(best_x)

        # Evaluate a few random points to seed the algorithm.
        # Candidate count must not exceed remaining budget.
        remaining = max_evals - evals_used
        if remaining > 0 and max_evals > 0:
            seed_count = min(self.pop_size - 1, remaining)
            # Ensure at least 1 seed if pop_size > 1
            for _ in range(seed_count):
                x = lo + np.random.rand(n) * (hi - lo)
                y = eval_point(x)
                if y < best_y:
                    best_y = y
                    best_x = x

        # If budget allows only initial evaluations, return.
        if evals_used >= max_evals:
            return best_x, best_y

        # If radius is degenerate, make it a small value based on bound spread.
        if radius < self.eps:
            # Use unit scale if span is zero everywhere (all points identical bounds)
            radius = 1.0

        # Iterative search
        best_prev = best_y
        stagnation = 0

        while evals_used < max_evals:
            remaining = max_evals - evals_used
            # Respect budget by choosing number of candidates this iteration.
            k = min(self.pop_size, remaining)
            if k <= 0:
                break

            # Determine exploitation vs exploration for each candidate.
            # Some candidates are uniform across bounds (exploration),
            # others are Gaussian around best_x (exploitation).
            # We also allow occasionally sampling the center itself.
            candidates = np.empty((k, n), dtype=float)

            # Noise scale per coordinate: proportional to span (if available)
            # with a fallback to a shared radius.
            # This makes sampling adapt to variable-wise bound scales.
            scale_vec = np.where(span > 0, span, 1.0)
            # Convert trust region radius to per-dimension standard deviation.
            # Using radius / sqrt(n) makes step magnitude roughly comparable across n.
            sigma = max(self.eps, radius / max(1.0, np.sqrt(n)))
            sigma_vec = sigma * (scale_vec / max(self.eps, float(np.mean(scale_vec))))

            for i in range(k):
                if np.random.rand() < self.explore_prob:
                    # Uniform exploration within bounds.
                    candidates[i] = lo + np.random.rand(n) * (hi - lo)
                else:
                    # Local exploitation around current best.
                    noise = np.random.randn(n) * sigma_vec
                    candidates[i] = best_x + noise

                # Optional: occasionally include the current best itself
                # to prevent losing it due to clipping.
                if k > 1 and np.random.rand() < 0.05:
                    candidates[i] = best_x

                # Boundary handling: clip to bounds.
                candidates[i] = np.minimum(np.maximum(candidates[i], lo), hi)

            # Evaluate all candidates this iteration (within budget).
            # If budget is tight, eval_point prevents exceeding, but we also keep k<=remaining.
            ys = np.empty(k, dtype=float)
            for i in range(k):
                ys[i] = eval_point(candidates[i])

            # Select the best among evaluated candidates
            idx = int(np.argmin(ys))
            y_new = float(ys[idx])
            x_new = candidates[idx]

            # Update global best
            if y_new < best_y:
                best_y = y_new
                best_x = x_new

            # Adapt radius based on improvement/stagnation
            if best_y < best_prev - 1e-12:
                # Progress: shrink radius to focus search.
                radius *= 0.85
                stagnation = 0
            else:
                # No progress: increase slightly and count stagnation.
                radius *= 1.05
                stagnation += 1

            best_prev = best_y

            # If stagnation persists, do a stronger random restart-like step
            # by expanding radius and injecting more exploratory candidates next iteration.
            if stagnation >= self.stagnation_limit:
                radius *= 2.0
                stagnation = 0
                # Make exploration more likely after stagnation.
                # Keep it bounded for stability.
                self.explore_prob = min(0.5, self.explore_prob + 0.05)
            else:
                # Slowly return exploration probability toward default.
                self.explore_prob = max(0.12, self.explore_prob - 0.01)

            # Safety: radius should not grow unbounded with infinite spans,
            # but bounds are typically finite. Still, clamp based on average span.
            # If span is zero, keep radius moderate.
            avg_span_now = float(np.mean(span)) if n > 0 else 0.0
            if avg_span_now > 0:
                radius = min(radius, 2.5 * avg_span_now + 1.0)
            else:
                radius = min(radius, 10.0)

        return best_x, best_y
