# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm combining
# surrogate-free local search (coordinate-wise finite-difference steps),
# random global exploration, and an adaptive step-size controller. It works
# for box-constrained domains in any dimension.
# Search state: Tracks current best solution x_best/y_best, a step scale
# sigma, and a simple archive of recent points (best + a few samples) to
# inform direction.
# Candidate generation: Each iteration proposes multiple candidates:
# (1) small local moves along coordinate-wise sign directions derived from
#     comparing function values at nearby perturbed points,
# (2) occasional isotropic random samples within the bounds,
# (3) candidate reflection/centering moves when local proposals hit bounds.
# Selection and replacement: Evaluates candidates and greedily keeps the best
# among current candidates as the new incumbent. Step size is updated based on
# whether an improvement was found.
# Adaptation: Uses a multiplicative step-size rule: sigma increases after
# successful improvement and decreases when improvements stall.
# Exploration mechanisms: Scheduled random sampling and occasional
# "restart-like" perturbations when no improvement occurs for a while.
# Exploitation mechanisms: Coordinate-wise local search around the current
# best using finite-difference-like sign tests to choose promising directions.
# Boundary handling: All candidate points are clipped to [lb, ub]. When a local
# direction would repeatedly go out of bounds, the algorithm nudges proposals
# towards the feasible region by using clipped moves and a centering term.
# Budget strategy: Converts the evaluation budget into a fixed maximum number of
# objective calls. Each iteration uses a bounded number of evaluations and the
# loop exits early if the budget would be exceeded.
# Closest known influences: Inspired by CMA-free evolution strategies and
# coordinate descent hybrids (finite-difference direction inference + adaptive
# step size), while remaining derivative-free and surrogate-free.
# Novelty or unusual aspects: Uses a lightweight coordinate-wise sign probing
# to infer promising move directions without requiring gradients, and combines
# that with bounded candidate pooling per iteration.
# Failure modes: If the objective is extremely noisy or has deceptive
# discontinuities, finite-difference sign inference may mislead; the random
# exploration and step-size decay mitigate but cannot fully prevent this.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # ---- Bounds ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # If bounds are not provided, assume a generic box.
            # (Still keep behavior robust; harness typically provides bounds.)
            lb = np.full(self.dim, -5.0, dtype=float)
            ub = np.full(self.dim, 5.0, dtype=float)

        lb = np.broadcast_to(lb, (self.dim,)).astype(float)
        ub = np.broadcast_to(ub, (self.dim,)).astype(float)
        span = ub - lb
        # Prevent zero-span dimensions from causing degenerate steps.
        span_safe = np.where(span > 0, span, 1.0)

        # ---- Budget guard ----
        max_evals = max(1, int(self.budget))
        evals = 0

        def eval_obj(x):
            nonlocal evals
            if evals >= max_evals:
                # Respect budget: if called too many times, just return current best.
                return best_y
            fx = func(x)
            evals += 1
            return float(fx)

        # ---- Initialization ----
        # Sample a few initial points to get a reasonable starting best.
        # Keep the number bounded by budget.
        init_tries = min(max_evals, 4 + self.dim // 2)
        # Start with the center point (often useful) then random points.
        x_center = lb + 0.5 * span_safe
        # Handle pathological span (e.g., all equal bounds):
        x_center = np.clip(x_center, lb, ub)

        # Evaluate center first
        best_x = x_center.copy()
        best_y = eval_obj(best_x)

        # Random initial points
        # Use uniform sampling within bounds.
        for _ in range(init_tries - 1):
            if evals >= max_evals:
                break
            r = np.random.rand(self.dim)
            x0 = lb + r * span_safe
            x0 = np.clip(x0, lb, ub)
            y0 = eval_obj(x0)
            if y0 < best_y:
                best_y = y0
                best_x = x0.copy()

        # Initial step size: fraction of typical span.
        # If spans are small, sigma should be small too.
        sigma = 0.25 * np.median(span_safe)
        sigma = float(sigma) if sigma > 0 else 0.1

        # Track failures for adaptive restart behavior.
        no_improve = 0
        success_in_a_row = 0

        # Archive of last best and a couple of samples (for centering and diversity).
        archive = [best_x.copy()]

        # ---- Main loop ----
        # Each iteration uses a small, budget-bounded number of evaluations.
        # We adapt the number of coordinate probes based on remaining budget.
        while evals < max_evals:
            remaining = max_evals - evals
            # We will spend at most this many evaluations this iteration.
            # Small constant + 1, scaled with dim but capped.
            # Coordinate probing costs ~2 per prob, so keep it modest.
            budget_this_iter = max(1, min(6 + self.dim // 2, remaining))

            # Determine how many coordinate directions to probe.
            # Each coordinate probe costs 2 evaluations: f(x+eps*e_i) and f(x-eps*e_i),
            # plus we'll also evaluate a couple random/local candidates. We'll keep within budget.
            # Choose k such that 2k + 2 <= budget_this_iter.
            k = min(self.dim, max(1, (budget_this_iter - 2) // 2))
            # Random subset of coordinates to probe (limits cost for large dim).
            coords = np.random.choice(self.dim, size=k, replace=False) if self.dim > k else np.arange(self.dim)

            eps = 1e-3
            # Scale eps with sigma and span to be dimensionless and robust.
            eps_vec = eps * np.maximum(span_safe, 1e-12)
            # Candidate pool (include current best as baseline).
            candidates = []
            cand_scores = []  # not used until evaluation
            # Always include a clipped "centered" move (helps escape when at boundaries).
            x_centered = best_x.copy()
            # Gentle pull towards the middle of the box (centering term).
            x_centered = 0.7 * x_centered + 0.3 * x_center
            x_centered = np.clip(x_centered, lb, ub)
            candidates.append(x_centered)

            # Isotropic exploration: one random move around best.
            # (Often helps in non-separable problems.)
            if remaining >= 3:
                z = np.random.randn(self.dim)
                # Normalize and scale
                zn = np.linalg.norm(z)
                if zn > 0:
                    z = z / zn
                x_rand = best_x + sigma * z
                x_rand = np.clip(x_rand, lb, ub)
                candidates.append(x_rand)

            # ---- Coordinate sign probing (exploitation) ----
            # Infer sign of improving direction along each prob coordinate by evaluating
            # two nearby perturbations; choose the better direction for that coordinate.
            # Then construct one aggregated local move from the coordinate signs.
            local_dirs = np.zeros(self.dim, dtype=float)
            # We'll evaluate both sides around the best for each prob coordinate.
            # Costs 2 evaluations per coordinate, capped by budget_this_iter via k.
            for i in coords:
                if evals >= max_evals:
                    break
                # Step for this coordinate
                step_i = eps_vec[i] + 0.01 * sigma * (span_safe[i] / (np.median(span_safe) + 1e-12))
                x_plus = best_x.copy()
                x_minus = best_x.copy()
                x_plus[i] = np.clip(x_plus[i] + step_i, lb[i], ub[i])
                x_minus[i] = np.clip(x_minus[i] - step_i, lb[i], ub[i])

                # If the plus and minus collapse to same point (tight bounds),
                # skip direction inference for this coordinate.
                if x_plus[i] == x_minus[i]:
                    continue

                y_plus = eval_obj(x_plus)
                y_minus = eval_obj(x_minus)
                # Choose direction that yields lower objective.
                if y_plus < y_minus:
                    local_dirs[i] = 1.0
                elif y_minus < y_plus:
                    local_dirs[i] = -1.0
                else:
                    local_dirs[i] = 0.0

                # Track incumbents from probing evaluations
                # (eval_obj updates best_y only indirectly; we update explicitly here)
                if y_plus < best_y:
                    best_y = y_plus
                    best_x = x_plus.copy()
                    archive.append(best_x.copy())
                if y_minus < best_y:
                    best_y = y_minus
                    best_x = x_minus.copy()
                    archive.append(best_x.copy())

            # Construct exploitation candidates using the inferred coordinate directions.
            if remaining > 0 and np.any(local_dirs != 0):
                # Aggregated coordinate move: best_x - sigma * dirs normalized on prob coords
                d = local_dirs.copy()
                # Normalize direction magnitude to avoid very large moves
                dn = np.linalg.norm(d)
                if dn > 0:
                    d /= dn
                x_local = best_x - sigma * d
                # Clip to bounds
                x_local = np.clip(x_local, lb, ub)
                candidates.append(x_local)

                # A second exploitation candidate: reflect around best (diversify in the same neighborhood)
                # Move in the opposite direction scaled slightly.
                x_local2 = best_x + 0.5 * sigma * d
                x_local2 = np.clip(x_local2, lb, ub)
                candidates.append(x_local2)

            # Additional bounded centering when many dimensions are at bounds:
            at_lower = best_x <= lb + 1e-12
            at_upper = best_x >= ub - 1e-12
            if np.count_nonzero(at_lower | at_upper) > self.dim // 2:
                # Nudge away from bounds by pulling towards center
                x_nudge = best_x + 0.25 * (x_center - best_x)
                x_nudge = np.clip(x_nudge, lb, ub)
                candidates.append(x_nudge)

            # ---- Evaluate candidate pool with strict budget adherence ----
            # Deduplicate candidates to reduce wasted evaluations.
            # Use rounding for stable deduplication.
            unique = []
            seen = set()
            for c in candidates:
                c = np.asarray(c, dtype=float)
                key = tuple(np.round(c, 12))
                if key not in seen:
                    seen.add(key)
                    unique.append(c)

            improved = False
            for x in unique:
                if evals >= max_evals:
                    break
                y = eval_obj(x)
                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                    archive.append(best_x.copy())
                    improved = True

            # Keep archive short
            if len(archive) > 6:
                archive = archive[-6:]

            # ---- Adapt sigma ----
            if improved:
                success_in_a_row += 1
                no_improve = 0
                # Increase sigma moderately after improvement to keep progress.
                sigma *= 1.15 if success_in_a_row >= 2 else 1.08
            else:
                no_improve += 1
                success_in_a_row = 0
                # Decrease sigma to refine.
                sigma *= 0.82

            # ---- Optional restart-ish perturbation ----
            # If stuck, do a larger random perturbation around the best.
            if no_improve >= 3 and evals < max_evals:
                # Spend at most 1 evaluation for this perturbation.
                z = np.random.randn(self.dim)
                zn = np.linalg.norm(z)
                if zn > 0:
                    z = z / zn
                # Larger step when stuck
                sigma_restart = max(sigma, 0.5 * np.median(span_safe))
                x_restart = best_x + sigma_restart * z
                x_restart = np.clip(x_restart, lb, ub)
                y = eval_obj(x_restart)
                if y < best_y:
                    best_y = y
                    best_x = x_restart.copy()
                    archive.append(best_x.copy())
                    no_improve = 0
                else:
                    # Mild additional shrink after a failed restart
                    sigma *= 0.9

            # If sigma becomes extremely small, nudge it to keep movement possible.
            min_sigma = 1e-12 * np.median(span_safe)
            if sigma < min_sigma and np.median(span_safe) > 0:
                sigma = min_sigma * 10.0

            # Safety break in case evaluation count stalls due to budget constraints.
            if evals >= max_evals:
                break

        return best_x, best_y
