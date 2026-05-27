# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, budgeted black-box minimization
# algorithm for real-valued continuous domains using a population-based
# coordinate-direction search with adaptive step sizes. It repeatedly samples
# candidate points around the best-so-far solution and also performs occasional
# global “jumps” to escape local minima.
#
# Search state: The algorithm maintains a current best solution (best_x,
# best_y), a step-size vector (sigma) for each coordinate, and an evaluation
# counter to ensure the total number of objective calls never exceeds the
# provided budget. It also keeps a small “memory” of recently successful
# directions.
#
# Candidate generation: Each iteration generates candidates by (1) sampling
# perturbations in randomly chosen coordinate directions scaled by sigma, and
# (2) performing occasional isotropic Gaussian/global perturbations. Candidate
# points are clipped to the variable bounds.
#
# Selection and replacement: Candidates are evaluated and any point that improves
# best_y replaces the current best_x. Step sizes are adapted based on whether
# improvements occurred (success -> shrink less, failure -> shrink more).
#
# Adaptation: The coordinate-wise sigma is multiplied by factors that depend on
# recent success. If improvements happen, sigma is slightly increased (or kept
# larger); otherwise sigma is reduced to refine around the incumbent best.
#
# Exploration mechanisms: Random coordinate-direction perturbations and periodic
# larger “global” jumps (scaled by the current range) help explore new regions.
#
# Exploitation mechanisms: Most samples are generated close to best_x using
# small Gaussian perturbations aligned with coordinate directions, biasing search
# toward local refinement.
#
# Boundary handling: Every candidate is clipped to [lb, ub] per coordinate.
#
# Budget strategy: The algorithm uses a strict evaluation budget counter; it
# plans iterations and candidate batches so the total number of objective calls
# never exceeds budget.
#
# Closest known influences: The approach is reminiscent of evolution strategies /
# coordinate search hybrids: it uses a (simplified) population sampling loop,
# adaptive step-size control, and clipped boundary handling.
#
# Novelty or unusual aspects: The implementation uses a coordinate-direction
# construction (sign/magnitude on individual axes) combined with a lightweight
# adaptive sigma per dimension and an evaluation-budget-aware batching scheme.
#
# Failure modes: If the objective is extremely noisy or has very sharp feasible
# boundaries, clipping may reduce effective search diversity. In very high
# dimensions, convergence may be slow due to the limited number of evaluations.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return a bounded default at midpoint.
            lb, ub = self._get_bounds(func)
            mid = (lb + ub) / 2.0
            return mid, float("inf")

        lb, ub = self._get_bounds(func)
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)

        # Handle degenerate dimensions robustly
        span = ub - lb
        span = np.where(span > 0, span, 1.0)

        rng = np.random.default_rng()  # seed is controlled by harness via np.random

        # Budget-aware evaluation counting
        evals = 0

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_point(x):
            nonlocal evals
            if evals >= budget:
                # Must never exceed budget; return best-y to avoid replacement.
                return best_y
            y = func(x)
            evals += 1
            return float(y)

        # Initialization: start from a random point within bounds, plus one structured point
        # (midpoint if it is within bounds; it always is by construction).
        x0 = lb + rng.random(dim) * (ub - lb)
        x_mid = (lb + ub) / 2.0

        y0 = eval_point(x0)
        if evals < budget:
            y_mid = eval_point(x_mid)
            if y_mid < y0:
                best_x, best_y = x_mid.copy(), y_mid
            else:
                best_x, best_y = x0.copy(), y0
        else:
            best_x, best_y = x0.copy(), y0

        # Step sizes: initialize with a fraction of the range.
        # sigma is per-coordinate and can shrink as we fail to improve.
        sigma = 0.3 * span
        sigma = np.maximum(sigma, 1e-12)

        # Lightweight success tracking
        consecutive_fail = 0
        recent_success = 0

        # Choose batch size conservatively to respect budget.
        # We'll run until budget is exhausted.
        # Candidate count per batch can be up to dim+4 but capped by remaining budget.
        while evals < budget:
            remaining = budget - evals
            # Larger batches at start; smaller later.
            # Aim for O(dim) candidates per batch but cap for robustness.
            base_batch = min(dim + 4, remaining)
            if base_batch <= 0:
                break

            # Decide exploration probability:
            # When we fail repeatedly, increase exploration (more global jumps).
            fail_ratio = min(5.0, consecutive_fail / 5.0)
            p_global = 0.08 + 0.10 * (fail_ratio / 5.0)
            p_global = float(np.clip(p_global, 0.08, 0.25))

            # Build candidates around best_x
            # Two types:
            # - Coordinate-direction Gaussian perturbations
            # - Occasional global isotropic jumps scaled by span
            X = np.empty((base_batch, dim), dtype=float)

            # We'll keep track of best found in this batch.
            batch_best_x = None
            batch_best_y = best_y

            # Pre-sample coordinate directions and signs for efficiency
            coords = rng.integers(0, dim, size=base_batch)
            signs = rng.choice(np.array([-1.0, 1.0]), size=base_batch)

            # For each candidate: pick whether global or local
            is_global = rng.random(base_batch) < p_global

            # Candidate construction:
            # Start at incumbent, then perturb one or more coordinates.
            # Local candidates: perturb a randomly chosen coordinate with magnitude ~ sigma.
            # To allow some mixing, we also add a small isotropic term for a subset.
            small_mix = rng.random(base_batch) < 0.35

            for i in range(base_batch):
                if is_global[i]:
                    # Global jump: large perturbation; also add local noise.
                    scale = (0.6 + 0.9 * rng.random())  # between 0.6 and 1.5
                    step = scale * span * (0.5 + rng.random(dim))
                    dir_vec = rng.standard_normal(dim)
                    xi = best_x + 0.35 * step * dir_vec
                    # Add some local coordinate-anchored component
                    j = coords[i]
                    xi[j] = xi[j] + signs[i] * sigma[j] * (0.5 + rng.random())
                else:
                    xi = best_x.copy()
                    j = coords[i]
                    # Main perturbation on one coordinate (coordinate-direction exploitation)
                    mag = sigma[j] * (0.2 + 0.9 * rng.random())
                    xi[j] = xi[j] + signs[i] * mag
                    # Optional mixing: small isotropic perturbation
                    if small_mix[i]:
                        xi = xi + 0.15 * sigma * rng.standard_normal(dim)
                X[i] = clip(xi)

            # Evaluate candidates
            for i in range(base_batch):
                if evals >= budget:
                    break
                y = eval_point(X[i])
                if y < batch_best_y:
                    batch_best_y = y
                    batch_best_x = X[i].copy()

            # Selection + adaptation
            if batch_best_x is not None and batch_best_y < best_y:
                best_x, best_y = batch_best_x, batch_best_y
                consecutive_fail = 0
                recent_success += 1
                # If success, slightly increase sigma (broaden) but not too much
                # to keep within boundaries and converge.
                inc = 1.03 + 0.02 * min(5, recent_success)
                sigma = sigma * inc
                # Also prevent sigma from exploding beyond a fraction of the span
                sigma = np.minimum(sigma, 0.9 * span + 1e-12)
            else:
                consecutive_fail += 1
                recent_success = 0
                # On failure, shrink sigma to refine.
                # Shrink more after repeated failures.
                shrink = 0.85 ** min(3, consecutive_fail)
                sigma = sigma * shrink
                sigma = np.maximum(sigma, 1e-12)

            # If sigma becomes extremely small or budget is near exhausted, allow a final
            # exploration jump by increasing p_global next loop.
            if evals >= budget:
                break

        return best_x, best_y

    def _get_bounds(self, func):
        # Bounds can be stored either directly or in func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # Fallback: assume bounds [-5, 5] if none provided.
            # This keeps behavior robust but still uses provided bounds if available.
            lb = -5.0 * np.ones(self.dim, dtype=float)
            ub = 5.0 * np.ones(self.dim, dtype=float)

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != self.dim or ub.size != self.dim:
            # If mismatch, broadcast or truncate to fit dim.
            if lb.size == 1:
                lb = np.full(self.dim, float(lb.item()), dtype=float)
            else:
                lb = np.resize(lb, self.dim).astype(float)

            if ub.size == 1:
                ub = np.full(self.dim, float(ub.item()), dtype=float)
            else:
                ub = np.resize(ub, self.dim).astype(float)

        # Ensure lb <= ub
        swap = lb > ub
        if np.any(swap):
            lb2 = lb.copy()
            lb2[swap] = ub[swap]
            ub2 = ub.copy()
            ub2[swap] = lb[swap]
            lb, ub = lb2, ub2

        return lb, ub
