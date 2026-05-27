# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# (derivative-free) based on a stochastic coordinate search with occasional
# global “restart-like” perturbations. It maintains a current best solution
# and iteratively samples small candidate moves around it, then replaces the
# incumbent if an improvement is found.
# Search state: The algorithm keeps (1) x_best and y_best, (2) a step size
# sigma controlling how far candidate points deviate from the current best,
# and (3) an evaluation counter to ensure the total number of objective calls
# never exceeds the provided budget.
# Candidate generation: At each iteration it creates a batch of candidates by
# adding random perturbations to x_best. Perturbations are mixtures of:
# (a) coordinate-aligned steps (random sign times per-dimension magnitude),
# (b) isotropic Gaussian steps, and (c) occasional larger jumps that help
# escape local minima when improvement stalls.
# Selection and replacement: It evaluates each candidate (within remaining
# budget). Any candidate with lower objective value than the current best
# becomes the new incumbent. If multiple improve, the best among them is used.
# Adaptation: The step size sigma is adapted using a simple success rule:
# sigma increases slightly after an improvement and decreases when an iteration
# produces no improvement.
# Exploration mechanisms: The occasional larger jumps (with probability that
# increases when improvements stall) provide exploration and reduce the risk of
# getting stuck in narrow basins.
# Exploitation mechanisms: Most candidate moves are small perturbations around
# x_best, focusing search effort near the current best.
# Boundary handling: All candidate points are clipped to the provided bounds.
# If bounds are not provided, it falls back to a safe default using [0, 1].
# Budget strategy: Each objective call decrements the remaining budget. The
# algorithm dynamically sizes candidate batches so it never exceeds the budget.
# Closest known influences: The approach is broadly inspired by stochastic
# direct-search and evolution-strategy-like step-size control, simplified
# for robustness and compactness.
# Novelty or unusual aspects: Uses a hybrid perturbation scheme combining
# coordinate-aligned and isotropic moves, plus a stall-adaptive larger-jump
# mechanism, while staying strictly budget-aware and seed-friendly.
# Failure modes: If the function is extremely noisy or highly discontinuous,
# sigma adaptation may oscillate or converge slowly. In very high dimensions,
# random exploration may require more budget to consistently find improvements.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = max(0, int(self.budget))

        # ---- Read bounds from func ----
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
            # Safe fallback: unit hypercube. (Harness should provide bounds.)
            lb = np.zeros(dim, dtype=float)
            ub = np.ones(dim, dtype=float)

        if lb.shape != (dim,):
            lb = np.broadcast_to(lb, (dim,)).astype(float)
        if ub.shape != (dim,):
            ub = np.broadcast_to(ub, (dim,)).astype(float)

        # Ensure numeric safety and valid ordering
        lb = np.minimum(lb, ub)
        ub = np.maximum(ub, lb + 0.0)

        span = ub - lb
        # Handle degenerate bounds (zero span): allow sigma to shrink to 0 in those dims
        span_safe = np.where(span > 0, span, 1.0)

        # ---- Budget-aware evaluation wrapper ----
        evals = 0

        def eval_x(x):
            nonlocal evals
            if evals >= budget:
                # Should not happen due to careful budgeting; return +inf to avoid use.
                return float("inf")
            x = np.asarray(x, dtype=float)
            # Clip defensively (callers may pass clipped already, but keep robust)
            x = np.minimum(np.maximum(x, lb), ub)
            y = func(x)
            evals += 1
            return float(y)

        # ---- Initialize incumbent ----
        # Start from a random point in the domain (seed controlled by harness).
        x_best = lb + np.random.rand(dim) * span_safe
        x_best = np.minimum(np.maximum(x_best, lb), ub)
        y_best = eval_x(x_best)

        if budget <= 1:
            return x_best, y_best

        # Initial step size: fraction of domain span.
        # Using median span for scale robustness across heterogeneous dimensions.
        scale0 = float(np.median(span_safe))
        # Avoid sigma=0 if all bounds are degenerate.
        sigma = 0.3 * scale0 if scale0 > 0 else 0.0

        # Stall counter controls exploration jumps.
        stall = 0
        max_no_improve = max(5, dim // 2)

        # Choose a base number of candidates per "iteration" without overshooting budget.
        # This keeps computations controlled under tight budgets.
        base_batch = 1 + min(24, dim)
        while evals < budget:
            remaining = budget - evals
            # Keep batch small enough to not exceed remaining evaluations
            batch = min(base_batch, remaining)

            # Probability of a larger jump increases with stall.
            # When stall is high, explore more aggressively.
            p_jump = 0.05 + 0.25 * (stall / max_no_improve)
            p_jump = float(min(0.35, max(0.0, p_jump)))

            improved = False
            x_best_prev = x_best.copy()
            y_best_prev = y_best

            # Generate candidates around current best.
            # Hybrid perturbations: coordinate-aligned + isotropic + occasional jumps.
            # Each candidate differs by its own random mask/sign and noise.
            for _ in range(batch):
                if sigma == 0.0:
                    # If sigma collapses, only attempt occasional random re-anchoring.
                    if np.random.rand() < p_jump:
                        x_cand = lb + np.random.rand(dim) * span_safe
                    else:
                        x_cand = x_best_prev
                    y_cand = eval_x(x_cand)
                    if y_cand < y_best:
                        x_best, y_best = np.asarray(x_cand, float), y_cand
                        improved = True
                    continue

                # Decide jump type
                if np.random.rand() < p_jump:
                    # Large jump: larger step proportional to span
                    # (increasing chance to escape local minima).
                    jump_factor = 2.0 + 4.0 * np.random.rand()
                    step_mag = sigma * jump_factor
                    # Mix isotropic with coordinate-aligned structure
                    z = np.random.randn(dim)
                    # Coordinate-aligned component
                    coord = np.random.randint(0, dim)
                    sign = 1.0 if np.random.rand() < 0.5 else -1.0
                    noise = z * 0.6
                    noise[coord] = sign * abs(np.random.randn()) * 1.5
                    x_cand = x_best_prev + step_mag * noise / np.sqrt(dim)
                else:
                    # Small step: exploit around incumbent.
                    # Use coordinate-aligned moves often to be effective in many problems.
                    # Magnitude per dimension scales with domain span.
                    if dim <= 0:
                        x_cand = x_best_prev
                    else:
                        # Coordinate-aligned
                        coord = np.random.randint(0, dim)
                        sign = 1.0 if np.random.rand() < 0.5 else -1.0
                        # Per-dimension magnitude (some dims may have tiny span).
                        mag = (sigma * (span_safe / np.median(span_safe))) if span_safe is not None else sigma
                        step = np.zeros(dim, dtype=float)
                        step[coord] = sign * abs(np.random.randn()) * mag[coord]
                        # Add a small isotropic component for coverage
                        step += (sigma * 0.25) * np.random.randn(dim) * (span_safe / np.median(span_safe))
                        x_cand = x_best_prev + step

                # Boundary handling via clipping
                x_cand = np.minimum(np.maximum(x_cand, lb), ub)
                y_cand = eval_x(x_cand)

                if y_cand < y_best:
                    x_best, y_best = np.asarray(x_cand, float), y_cand
                    improved = True

                # Early exit from candidate generation if we already improved;
                # still allow remaining candidates to potentially find even better.
                # (Keeps behavior simple; not strict early stopping.)

            # ---- Adapt sigma based on success ----
            if improved:
                stall = 0
                # Increase sigma slightly to exploit improved region more effectively.
                # Cap sigma to avoid jumping out too far.
                sigma = sigma * (1.05 + 0.05 * np.random.rand())
                sigma = min(sigma, 1.5 * float(np.max(span_safe)))
            else:
                stall += 1
                # Decrease sigma after no improvement
                sigma = sigma * 0.85
                # If sigma gets too small and we stall, reset exploration
                if stall >= max_no_improve or sigma < 1e-12:
                    stall = 0
                    # Reset sigma to a reasonable fraction of the domain span.
                    sigma = 0.3 * float(np.median(span_safe))
                    # Random re-anchoring to escape stagnation, but stay within budget.
                    if evals < budget:
                        x_restart = lb + np.random.rand(dim) * span_safe
                        y_restart = eval_x(x_restart)
                        if y_restart < y_best:
                            x_best, y_best = x_restart, y_restart

            # Protect against pathological sigma collapse/degeneracy
            if evals >= budget:
                break

            # If objective improved but sigma is zero due to degenerate span,
            # keep incumbent and continue with no-op moves (budget will end).
            if np.all(span == 0):
                break

        return np.asarray(x_best, dtype=float), float(y_best)
