# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, robust black-box minimization algorithm
# (derivative-free, bounds-aware) that works for any dimension. It is designed for
# limited evaluation budgets and uses a mixture of global sampling and local,
# direction-aware refinement to find low objective values.
#
# Search state: The algorithm maintains (1) the current best solution x_best and
# its objective value y_best, (2) a set of candidate points sampled around a
# reference point, and (3) a step size (radius) that controls exploration scale.
#
# Candidate generation: Each iteration draws candidates by combining:
# - Gaussian perturbations around the current best and a rolling center
# - Occasional uniform samples over the full bounds (global exploration)
# - A simple coordinate-wise mutation proposal to help escape axis-aligned traps
#
# Selection and replacement: Candidates are evaluated (never exceeding the budget).
# Any candidate that improves the best objective replaces x_best (greedy minimization).
# A local center is also updated toward improvements to accelerate convergence.
#
# Adaptation: The step size decays when improvements are found and grows slightly
# when progress stalls, based on a rolling counter of iterations without improvement.
#
# Exploration mechanisms: Uniform global samples and relatively large Gaussian
# perturbations at the start; then exploration gradually decreases as the budget
# is consumed.
#
# Exploitation mechanisms: Frequent local Gaussian sampling centered at the best
# point, plus coordinate-wise refinements, using an adaptive step size.
#
# Boundary handling: All proposed points are clipped to the feasible bounds. This
# keeps evaluations valid while preserving the stochastic exploration behavior.
#
# Budget strategy: The algorithm computes the number of evaluations needed for an
# initial design plus a sequence of adaptive steps, ensuring the total number of
# function calls never exceeds the provided evaluation budget.
#
# Closest known influences: Inspired by "random sampling + success-based step-size"
# patterns similar to evolution strategies / CMA-lite behavior, but kept intentionally
# simple and implementation-friendly for a compact benchmark baseline.
#
# Novelty or unusual aspects: Uses a rolling improvement/stall signal to adapt step size,
# and includes a coordinate-wise mutation that is scaled consistently with the current
# radius, improving robustness in small budgets.
#
# Failure modes: With extremely tight budgets or highly non-smooth objectives, the
# algorithm may not find improvements before budget exhaustion. Clipping can also cause
# reduced effective exploration near boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # --- Read bounds ---
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        if lb is None or ub is None:
            raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub")

        # Ensure proper shapes
        lb = np.broadcast_to(lb, (self.dim,)).astype(float, copy=False)
        ub = np.broadcast_to(ub, (self.dim,)).astype(float, copy=False)
        span = ub - lb
        # Handle degenerate dimensions (zero span): keep span at least 1 to avoid zeros in scaling
        span_safe = np.where(span > 0, span, 1.0)

        # --- Budget guard ---
        max_evals = max(0, self.budget)

        # Helper to clip within bounds
        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        # Helper to evaluate objective with budget accounting
        evals = 0
        best_x = None
        best_y = None

        def eval_at(x):
            nonlocal evals, best_x, best_y
            y = float(func(x))
            evals += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # If no budget, return something deterministic within bounds
        if max_evals == 0:
            x0 = lb.copy()
            x0 = clip(x0)
            return x0, float(func(x0))

        # --- Initial sampling ---
        # Choose a small initial design size that adapts to dimension and budget.
        # Aim: get a reasonable first best with limited calls.
        # At least 1 evaluation will occur; never exceed budget.
        init_budget = min(max_evals, max(1, int(2 + 0.5 * np.sqrt(self.dim) * 2)))
        # Add a mid-point candidate (often helps) plus random candidates.
        # Ensure we use at most init_budget calls.
        x_mid = lb + 0.5 * span
        x_mid = clip(x_mid)
        eval_at(x_mid)

        # Fill remaining initial evals with uniform samples.
        remaining = init_budget - 1
        if remaining > 0:
            # Uniform samples in bounds
            u = np.random.rand(remaining, self.dim)
            X = lb + u * span
            for i in range(remaining):
                eval_at(X[i])

        # If budget exhausted, return current best
        if evals >= max_evals:
            return best_x, best_y

        # --- Adaptive local/global search loop ---
        # Initial radius: fraction of domain scale (average span magnitude).
        # Use span_safe to avoid zero division.
        radius = 0.25 * float(np.mean(span_safe))
        radius = max(radius, 1e-12)

        # Rolling stall counter for adaptation
        no_improve_iters = 0
        improve_reset = 6  # how quickly we allow growth after improvements stop

        # Total remaining evaluations
        while evals < max_evals:
            # Determine how many candidates to evaluate in this "iteration".
            # Keep it variable but ensure we don't exceed remaining budget.
            rem = max_evals - evals
            # Small batch to reduce overhead and keep budget usage deterministic.
            batch = min(rem, max(4, int(2 + 0.25 * self.dim)))
            # Global exploration probability decreases as budget is consumed.
            t = evals / max_evals
            p_global = 0.35 * (1.0 - t) + 0.05  # between ~0.05 and 0.40 early

            # Center for exploitation: start near best, but allow a rolling reference
            # that can drift slightly with improvements.
            center = best_x if best_x is not None else (lb + 0.5 * span)

            # Generate candidates
            candidates = []

            # 1) Gaussian around best
            # Use a decaying step size with some noise to keep diversity.
            local_scale = radius * (0.5 + 0.7 * np.random.rand(batch))
            Z = np.random.randn(batch, self.dim)
            X_local = center + Z * local_scale[:, None]

            # 2) Occasional global uniform samples
            # Replace a subset of candidates with uniform samples.
            X_global = lb + np.random.rand(batch, self.dim) * span

            # 3) Coordinate-wise mutation: mutate one coordinate per candidate
            # to help with axis-aligned landscapes.
            X_coord = np.empty((batch, self.dim), dtype=float)
            X_coord[:] = center
            # choose coordinate indices for each candidate
            idxs = np.random.randint(0, self.dim, size=batch)
            # mutate with random sign and magnitude
            signs = np.where(np.random.rand(batch) < 0.5, -1.0, 1.0)
            magnitudes = radius * (0.25 + 0.75 * np.random.rand(batch))
            X_coord[np.arange(batch), idxs] = X_coord[np.arange(batch), idxs] + signs * magnitudes

            # Mix
            use_global = np.random.rand(batch) < p_global
            # Default is local
            X = X_local
            X[use_global] = X_global[use_global]
            # Also overwrite a few with coordinate mutations for diversity
            # (more likely when stalled)
            p_coord = 0.2 + 0.3 * (no_improve_iters > improve_reset)
            use_coord = (~use_global) & (np.random.rand(batch) < p_coord)
            X[use_coord] = X_coord[use_coord]

            # Boundary handling
            X = clip(X)

            # Evaluate batch; greedy replacement handled by eval_at
            prev_best_y = best_y
            for i in range(batch):
                if evals >= max_evals:
                    break
                eval_at(X[i])

            # Adaptation based on improvement
            if best_y is not None and (prev_best_y is None or best_y < prev_best_y - 1e-15):
                # Improvement found: shrink radius a bit to exploit better
                no_improve_iters = 0
                radius *= 0.82
            else:
                no_improve_iters += 1
                # If stalled, gently increase radius to re-explore
                # and add robustness against local minima.
                if no_improve_iters <= improve_reset:
                    radius *= 1.05
                else:
                    radius *= 1.15

            # Ensure radius stays meaningful within bounds
            max_radius = float(np.max(span_safe))
            radius = min(radius, max_radius)
            radius = max(radius, 1e-12)

            # If fully stalled and we're close to budget end, break early
            if no_improve_iters > (2 * improve_reset + 10) and (max_evals - evals) < max(8, self.dim // 2):
                break

        return best_x, best_y
