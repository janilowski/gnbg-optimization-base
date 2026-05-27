# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm using a
# derivative-free, evolution-strategy style search. It maintains a small
# population of candidate solutions and iteratively refines a “center”
# point toward better objective values while controlling step sizes.
#
# Search state: Tracks remaining function evaluations, an incumbent best
# (x_best, y_best), a population of candidate points, and an adaptive
# Gaussian step size (sigma) plus a mutation scale. Also tracks a
# restart counter when progress stalls.
#
# Candidate generation: In each iteration, new candidates are created by
# sampling around the current center using isotropic Gaussian noise:
#   x_new = x_center + sigma * z, where z ~ N(0, I).
# The center updates to the best candidates, and sigma adapts based on
# whether improvement occurred.
#
# Selection and replacement: Among offspring, selects the top individuals
# (lowest objective values). The next generation’s center becomes a weighted
# recombination (closer to the best), while the rest of the population is
# replaced by newly sampled points around the updated center.
#
# Adaptation: If the algorithm finds a better solution than the incumbent,
# sigma is reduced (local refinement). If not, sigma is increased or a
# restart is triggered after repeated stagnation, improving robustness.
#
# Exploration mechanisms: Larger sigma during stagnation and random
# restarts spread sampling across the space to escape local minima.
#
# Exploitation mechanisms: After improvements, sigma shrinks to focus
# sampling near the best-known region. Weighted recombination further
# concentrates search around high-quality candidates.
#
# Boundary handling: After sampling, candidates are clipped into the
# provided bounds. This keeps the search feasible without requiring
# function-specific repair operators.
#
# Budget strategy: Uses a strict evaluation counter; every call to the
# objective is counted, and the algorithm stops when the budget is exhausted
# (never exceeding the provided evaluation budget).
#
# Closest known influences: A simplified (mu+lambda)-ES / CMA-like “center
# search” variant: elitist selection, recombination, adaptive step size, and
# stagnation-based restarts.
#
# Novelty or unusual aspects: Very compact implementation with careful budget
# accounting and generic handling of bounds through multiple possible
# attributes (func.lower/upper or func.bounds.lb/ub). Uses isotropic sigma
# adaptation rather than full covariance learning.
#
# Failure modes: If the objective is extremely noisy or bounds are very
# tight relative to sigma, progress may stall; restarts mitigate but cannot
# guarantee success under adversarial noise or pathological objectives.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = max(0, int(self.budget))

        # ---- Bounds handling (try multiple attribute paths) ----
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        else:
            b = getattr(func, "bounds", None)
            if b is None:
                raise AttributeError("func must provide bounds via lower/upper or bounds.lb/bounds.ub")
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)

        # Broadcast / validate to correct dimension
        if lb.size == 1:
            lb = np.full(dim, float(lb))
        if ub.size == 1:
            ub = np.full(dim, float(ub))
        if lb.shape != (dim,) or ub.shape != (dim,):
            raise ValueError(f"Bounds must be scalar or shape ({dim},); got lb{lb.shape}, ub{ub.shape}")

        # Ensure finite bounds (if user gives inf, clipping becomes problematic)
        # We still attempt to proceed; sigma and clipping will behave, but infinities
        # make clipping ineffective. This is user responsibility.
        span = ub - lb
        span = np.where(span == 0, 1.0, span)  # avoid divide-by-zero in scaling

        # ---- Evaluation wrapper with strict budget accounting ----
        evals = 0

        # The objective is minimization.
        def eval_obj(x):
            nonlocal evals
            if evals >= budget:
                # Must not exceed budget; return incumbent-like large value.
                # But in practice, we guard loops so this shouldn't happen.
                return np.inf
            evals += 1
            y = func(np.asarray(x, dtype=float))
            # Some black-boxes may return non-float; coerce robustly
            return float(y)

        if budget == 0:
            # Can't evaluate; return a feasible point
            x0 = np.clip(np.zeros(dim), lb, ub)
            return x0, float("inf")

        # ---- Initialization ----
        rng = np.random

        # Choose initial center: mid-point (robust) + small noise if possible.
        x_mid = 0.5 * (lb + ub)
        # Initial sigma: fraction of span (clamped to avoid numerical issues).
        # Use span scale to be dimension-agnostic.
        sigma = 0.3 * np.mean(np.abs(span)) / (1.0 + 0.1 * np.log1p(dim))
        sigma = float(max(1e-12, sigma))

        # Population size and offspring count chosen based on budget.
        # Aim for 1..8-ish candidates per step depending on budget and dimension.
        # We'll perform "generations" of size pop_size by sampling pop_size points
        # around the center each generation.
        pop_size = int(np.clip(2 + dim // 5, 2, 10))
        pop_size = min(pop_size, max(2, budget))  # can't exceed budget in practice

        # Stagnation control
        best_y = float("inf")
        best_x = None

        # Evaluate an initial candidate at the center (counts to budget)
        y0 = eval_obj(x_mid)
        best_y = y0
        best_x = x_mid.copy()

        # If remaining budget is 0, return early
        if evals >= budget:
            return best_x, best_y

        # Start with a small initial population around the best center
        center = best_x.copy()
        stagnation = 0
        restart_limit = 6  # number of consecutive stagnations before restarting more aggressively

        # ---- Main loop ----
        # Each generation samples pop_size offspring and uses elitist selection.
        # Keep it simple and robust; never evaluate beyond budget.
        while evals < budget:
            # Offspring generation
            # Use isotropic Gaussian noise; then clip to bounds.
            # Generate in batches but still using single eval wrapper for budget.
            candidates = np.empty((pop_size, dim), dtype=float)
            ys = np.empty(pop_size, dtype=float)

            # Adaptive mutation scale: can inflate when sigma is very small.
            # Also uses a gentle dependence on dimension.
            base_scale = sigma
            z_scale = 1.0 / np.sqrt(dim)  # keep magnitude roughly stable with dim
            # For stability, ensure at least a tiny movement range.
            min_step = 1e-12 * np.mean(np.abs(span))
            base_scale = float(max(min_step, base_scale))

            for i in range(pop_size):
                if evals >= budget:
                    break

                # Sample mutation
                z = rng.randn(dim)
                x = center + (base_scale * z_scale) * z

                # Boundary handling: clip into feasible region
                x = np.minimum(np.maximum(x, lb), ub)

                candidates[i] = x
                ys[i] = eval_obj(x)

            # If we broke early due to budget, resize arrays logically
            valid_n = min(pop_size, len(ys))
            # Note: if evals ran out inside the loop, some ys might be uninitialized.
            # Detect valid portion by comparing against inf due to eval_obj guard (or just
            # use eval count vs iterations). We'll be conservative: rebuild selection
            # using finite values only.
            finite_mask = np.isfinite(ys)
            if not np.any(finite_mask):
                break

            candidates = candidates[finite_mask]
            ys = ys[finite_mask]
            valid_n = ys.size

            # Selection: choose elites with smallest objective values.
            # Use k = min(3, valid_n) to keep it compact.
            k = min(3, valid_n)
            elite_idx = np.argpartition(ys, kth=k - 1)[:k]
            elite_x = candidates[elite_idx]
            elite_y = ys[elite_idx]

            # Update incumbent best
            best_i = int(np.argmin(elite_y))
            if elite_y[best_i] < best_y:
                best_y = float(elite_y[best_i])
                best_x = elite_x[best_i].copy()
                improved = True
            else:
                improved = False

            # Recombination to update the center:
            # Weighted by relative quality (lower y => higher weight).
            # Add small epsilon to avoid division by zero.
            # Use a bounded softmax-like weighting:
            # w_i ∝ exp(-(y - y_min)/scale) with scale tied to spread.
            y_min = float(np.min(elite_y))
            y_spread = float(np.max(elite_y) - y_min)
            scale = y_spread + 1e-12
            # Scale down exponent if spread is tiny
            denom = scale
            ex = np.exp(-(elite_y - y_min) / denom)
            w = ex / (np.sum(ex) + 1e-12)
            center = np.sum(elite_x * w[:, None], axis=0)

            # Adapt sigma based on success
            if improved:
                stagnation = 0
                # Shrink sigma for exploitation
                sigma *= 0.8
            else:
                stagnation += 1
                # If no improvement, expand sigma modestly (exploration)
                sigma *= 1.08

            # Restart mechanism: if too many stagnations, reset center randomly.
            # Ensure we still use bounds and feasibility.
            if stagnation >= restart_limit and evals < budget:
                stagnation = 0
                # Randomly reinitialize around uniform samples in bounds
                # plus a slight bias toward current best_x.
                if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
                    x_rand = lb + rng.rand(dim) * (ub - lb)
                else:
                    # If bounds contain inf, fall back to best_x + noise
                    x_rand = center + rng.randn(dim) * (0.5 * sigma)

                # Bias toward current best for stability
                center = 0.7 * best_x + 0.3 * x_rand
                center = np.minimum(np.maximum(center, lb), ub)

                # Reset sigma to a broader scale
                sigma = 0.35 * np.mean(np.abs(span)) / (1.0 + 0.05 * np.log1p(dim))
                sigma = float(max(1e-12, sigma))

            # Additional early termination: if sigma is extremely small, it may not move.
            if sigma < 1e-12 * np.mean(np.abs(span)):
                # Still allow a couple more loops by slightly increasing.
                sigma = max(sigma, 1e-12) * 10.0

        # Final safeguard: ensure best_x is feasible
        if best_x is None:
            best_x = np.clip(np.zeros(dim), lb, ub)
            best_y = eval_obj(best_x) if evals < budget else float("inf")
        else:
            best_x = np.minimum(np.maximum(best_x, lb), ub)

        return best_x, float(best_y)
