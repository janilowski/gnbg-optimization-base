# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimization algorithm
# suitable for bounded continuous domains using a hybrid of coordinate/gradient-free
# search and adaptive stochastic restarts. It uses only function evaluations and
# respects the provided evaluation budget.
# Search state: Maintains a current best solution (x_best, y_best), a step size
# sigma controlling mutation scale, and an optional “direction memory” based on
# successful coordinate improvements. Also keeps track of evaluation count.
# Candidate generation: Generates new candidate points around the current best
# via (1) Gaussian perturbations scaled by sigma, (2) coordinate probes
# along remembered improving directions, and (3) occasional random restarts
# to escape stagnation.
# Selection and replacement: Each generated candidate is evaluated; if it improves
# the best observed value, it replaces x_best and y_best. Direction memory is
# updated when a coordinate probe yields improvement.
# Adaptation: sigma increases slightly after repeated failures (to re-explore) and
# decreases after improvements (to refine). Restart logic triggers when no
# improvement occurs for a fraction of the budget.
# Exploration mechanisms: Random Gaussian mutations and periodic full random
# restart; coordinate probes encourage structured exploration in useful directions.
# Exploitation mechanisms: Shrinking sigma after improvements and using remembered
# directions for coordinate probes around the current best.
# Boundary handling: Uses a projection (clipping) strategy to keep candidates within
# bounds after mutation/probing. Bounds are read from func.lower/func.upper or
# func.bounds.lb/func.bounds.ub.
# Budget strategy: Uses exactly the provided evaluation budget (or less only if
# bounds are invalid). The algorithm stops when the budget is exhausted.
# Closest known influences: Inspired by simple evolution strategies (1+lambda) and
# pattern search hybrids, with adaptive step size and bounded projection.
# Novelty or unusual aspects: Mixes stochastic isotropic mutations with adaptive
# coordinate probes and a direction-memory update derived from recent improvements.
# Failure modes: If the objective is extremely noisy or highly discontinuous,
# pure improvement-based adaptation can stagnate; the periodic restarts mitigate
# this but cannot guarantee performance.
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

        # --- Read bounds from the function object ---
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
            # Fallback: assume [0, 1]^dim if bounds are unavailable
            lb = np.zeros(dim, dtype=float)
            ub = np.ones(dim, dtype=float)
        if lb.shape[0] != dim or ub.shape[0] != dim:
            lb = np.resize(lb, dim).astype(float, copy=False)
            ub = np.resize(ub, dim).astype(float, copy=False)

        # Ensure proper ordering
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo
        span[span == 0.0] = 1.0  # avoid degenerate scaling; will clip anyway

        def project(x):
            return np.clip(x, lo, hi)

        # --- Objective wrapper with budget control ---
        evals = 0

        def f(x):
            nonlocal evals
            if evals >= budget:
                return np.inf
            x = np.asarray(x, dtype=float)
            y = float(func(x))
            evals += 1
            return y

        # --- Initialization ---
        rng = np.random

        # Start from a random feasible point
        x_best = project(lo + rng.rand(dim) * span)
        y_best = f(x_best)

        # Step size: proportional to domain size
        sigma = 0.3 * np.mean(span)
        sigma = float(max(sigma, 1e-12))

        # Direction memory: per-coordinate weights (used for coordinate probes)
        dir_weights = np.zeros(dim, dtype=float)
        # Keep a sign-less memory; probe uses positive/negative randomly
        no_improve = 0

        # How many candidates per "round"
        # Keep it small to stay compact and flexible with dimension.
        lambda_ = int(4 + 3 * np.log1p(dim))
        lambda_ = max(6, lambda_)
        lambda_ = min(lambda_, budget)  # won't exceed budget overall anyway

        # Restart schedule
        # Trigger restart if no improvement for a fraction of budget.
        restart_patience = max(10, int(0.08 * budget))
        # Shrink schedule
        shrink_after_improve = 0.85
        grow_after_fail = 1.08

        # Coordinate probe probability
        p_coord = 0.35
        # Random restart probability inside a round
        p_restart = 0.10

        # --- Main loop ---
        while evals < budget:
            improved_in_round = False

            # A "round" creates up to lambda_ candidates or until budget is reached.
            for _ in range(lambda_):
                if evals >= budget:
                    break

                # Decide candidate type
                use_coord = (rng.rand() < p_coord) and (np.any(dir_weights != 0.0))
                use_restart = (rng.rand() < p_restart) and (no_improve >= restart_patience)

                if use_restart:
                    # Full restart: sample a new point; keep sigma moderately sized.
                    x_cand = project(lo + rng.rand(dim) * span)
                    y_cand = f(x_cand)

                    # Reset memory a bit on restart; but keep global best.
                    if y_cand < y_best:
                        x_best, y_best = x_cand, y_cand
                        improved_in_round = True
                        no_improve = 0
                        sigma = max(sigma * shrink_after_improve, 1e-12)
                    else:
                        no_improve += 1
                        sigma = min(sigma * grow_after_fail, 0.8 * np.mean(span) + 1e-12)
                    continue

                if use_coord:
                    # Coordinate probe along a weighted coordinate.
                    # Choose top-weighted coordinate with some randomness.
                    weights = dir_weights
                    idx_pool = np.argsort(weights)[::-1]
                    k = idx_pool[: max(2, min(6, dim))]
                    j = int(k[rng.randint(0, len(k))])

                    # Step along coordinate j with random sign.
                    step = (sigma * (0.8 + 0.6 * rng.rand())) * rng.choice([-1.0, 1.0])
                    x_cand = x_best.copy()
                    x_cand[j] = x_cand[j] + step
                    x_cand = project(x_cand)
                    y_cand = f(x_cand)

                    if y_cand < y_best:
                        x_best, y_best = x_cand, y_cand
                        improved_in_round = True
                        no_improve = 0

                        # Strengthen the remembered coordinate direction.
                        # Keep it bounded to avoid runaway weights.
                        dir_weights[j] = min(5.0, dir_weights[j] + 0.25 + 0.25 * rng.rand())

                        sigma = max(sigma * shrink_after_improve, 1e-12)
                    else:
                        no_improve += 1
                        # Light decay of the coordinate's weight on failure
                        dir_weights[j] *= 0.98
                        sigma = min(sigma * grow_after_fail, 0.9 * np.mean(span) + 1e-12)

                else:
                    # Isotropic Gaussian mutation around best.
                    # Use both Gaussian and occasional heavy-tailed step.
                    if rng.rand() < 0.2:
                        # Heavy-tail from Laplace for occasional large moves
                        noise = rng.laplace(loc=0.0, scale=1.0, size=dim)
                    else:
                        noise = rng.normal(loc=0.0, scale=1.0, size=dim)

                    x_cand = x_best + sigma * noise
                    x_cand = project(x_cand)
                    y_cand = f(x_cand)

                    if y_cand < y_best:
                        x_best, y_best = x_cand, y_cand
                        improved_in_round = True
                        no_improve = 0

                        # Adapt: shrink and also (weakly) update direction memory
                        # based on which coordinates moved most in the successful candidate.
                        delta = x_cand - x_best
                        if np.any(delta != 0.0):
                            coord_mag = np.abs(delta) / (np.abs(span) + 1e-12)
                            # Update top movers
                            j = int(np.argmax(coord_mag))
                            if np.isfinite(coord_mag[j]) and coord_mag[j] > 0:
                                dir_weights[j] = min(5.0, dir_weights[j] + 0.15 + 0.35 * rng.rand())

                        sigma = max(sigma * shrink_after_improve, 1e-12)
                    else:
                        no_improve += 1
                        sigma = min(sigma * grow_after_fail, 0.9 * np.mean(span) + 1e-12)

                if evals >= budget:
                    break

            # If no improvement in a round, decay coordinate memory slightly.
            if not improved_in_round:
                dir_weights *= 0.99
                # Extra sigma growth if stagnating and not near budget end
                if no_improve > restart_patience and evals < budget:
                    sigma = min(sigma * 1.12, 0.95 * np.mean(span) + 1e-12)

            # If budget is very small, break cleanly.
            if evals >= budget:
                break

        # Ensure best_x is within bounds
        x_best = project(x_best)
        return x_best, y_best
