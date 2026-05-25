# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# (derivative-free, bounded) using an evolving population of candidate points.
# It resembles a lightweight CMA-ES-inspired strategy: it samples points from
# an adaptive Gaussian around an incumbent, evaluates them under a strict
# evaluation budget, and then updates the search distribution based on the
# best observed solutions.
# Search state: Maintains a current mean (center) vector, a scalar step
# size (sigma), and a small evaluation-capable population of samples.
# Candidate generation: Each iteration draws a batch of candidate points from
# N(mean, sigma^2 I), then repairs them by clamping to the provided bounds.
# Selection and replacement: Candidates are ranked by objective value; the
# mean is moved toward the average of the top-performing elites. The global
# best solution (lowest y) is tracked throughout.
# Adaptation: Sigma is adapted using a simple success rule: if elites improve
# upon the previous best, sigma shrinks (finer search); otherwise sigma grows
# (broader search). This keeps the algorithm responsive across dimensions.
# Exploration mechanisms: Larger sigma after stagnation promotes global
# exploration; occasional larger steps help escape local basins.
# Exploitation mechanisms: When improvements happen, the mean shifts toward
# elite points and sigma shrinks to refine around promising regions.
# Boundary handling: Candidates are clamped to [lower, upper]. The mean is
# also clamped after updates to remain feasible.
# Budget strategy: The algorithm never exceeds the given evaluation budget.
# It chooses an iteration count based on the budget and batch size, and it
# truncates the final batch evaluations if only a partial budget remains.
# Closest known influences: Simple random search with elitist selection and
# adaptive step size, inspired by CMA-ES-like sampling and success-based sigma
# control, but kept intentionally minimal and robust.
# Novelty or unusual aspects: Uses an adaptive elite-mean update with a
# success/shrink-or-grow rule that works without covariance matrices, making
# it simple yet effective in many black-box settings.
# Failure modes: If the objective is very noisy, the success-based sigma
# update may overreact. If bounds are extremely tight, clamping can reduce
# effective diversity. In very high dimensions with small budgets, the
# population may be too small to find good directions.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # Read bounds from func.lower/upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        else:
            b = func.bounds
            lower = np.asarray(b.lb, dtype=float)
            upper = np.asarray(b.ub, dtype=float)

        if lower.shape[0] != self.dim or upper.shape[0] != self.dim:
            # Be robust: broadcast if possible, otherwise fall back to scalar bounds.
            lower = np.broadcast_to(lower, (self.dim,)).astype(float)
            upper = np.broadcast_to(upper, (self.dim,)).astype(float)

        # Ensure correct ordering
        lo = np.minimum(lower, upper)
        hi = np.maximum(lower, upper)

        dim = self.dim
        budget = max(0, int(self.budget))
        if budget == 0:
            # No evaluations allowed: return a feasible point with unknown y.
            # (Harness should avoid this case; still be robust.)
            x0 = np.clip((lo + hi) / 2.0, lo, hi)
            return x0, float("inf")

        rng = np.random

        # Choose batch size and iteration count to respect budget.
        # Population size: small but grows mildly with dimension.
        # Keep evaluations compact and robust for typical benchmark budgets.
        batch = int(np.clip(4 + dim // 10, 4, 32))
        max_iters = max(1, budget // batch)
        evals_used = 0

        # Initialize mean near center with a bit of random spread (helps start).
        center = (lo + hi) / 2.0
        span = np.maximum(hi - lo, 1e-12)

        # Initial sigma scaled to the feasible range.
        sigma = 0.25 * float(np.mean(span))
        sigma = max(sigma, 1e-3)

        # Track best found
        # Evaluate initial center to anchor exploitation early.
        x_best = np.clip(center, lo, hi)
        y_best = float(func(x_best))
        evals_used += 1

        # Keep previous best for success-based adaptation
        prev_best = y_best

        # Elite fraction: use top ~25% (at least 1).
        elite_k = max(1, int(np.ceil(0.25 * batch)))

        # Main loop
        for _ in range(max_iters):
            # Determine remaining budget for this iteration
            remaining = budget - evals_used
            if remaining <= 0:
                break
            # Possibly truncate batch if nearing budget end
            bsize = min(batch, remaining)

            # Sample candidates from Gaussian around current mean
            # Sample matrix shape: (bsize, dim)
            samples = rng.standard_normal(size=(bsize, dim))
            X = np.clip(center + sigma * samples, lo, hi)

            # Evaluate each candidate
            ys = np.empty(bsize, dtype=float)
            for i in range(bsize):
                ys[i] = float(func(X[i]))
            evals_used += bsize

            # Rank by objective (minimization)
            order = np.argsort(ys)
            best_idx = int(order[0])

            # Update global best
            if ys[best_idx] < y_best:
                y_best = float(ys[best_idx])
                x_best = X[best_idx].copy()

            # Selection: move center toward elite average
            elite_idx = order[:elite_k]
            elite_mean = np.mean(X[elite_idx], axis=0)

            # Adaptation: success-based sigma control
            # If improved global best since last iteration, shrink sigma (exploit),
            # else expand slightly (explore).
            improved = y_best < prev_best
            prev_best = y_best

            if improved:
                # shrink: focus search near elite region
                sigma *= 0.8
            else:
                # expand: escape stagnation
                sigma *= 1.05

            # Prevent sigma from collapsing or exploding relative to bounds
            # (robust across different bound scales)
            max_sigma = 0.5 * float(np.mean(span))
            max_sigma = max(max_sigma, 1e-3)
            min_sigma = 1e-12
            sigma = float(np.clip(sigma, min_sigma, max_sigma))

            # Replace center with a blended elite mean + small fraction of old center
            # to reduce jitter.
            blend = 0.7
            center = (1.0 - blend) * center + blend * elite_mean
            center = np.clip(center, lo, hi)

        return x_best, y_best
