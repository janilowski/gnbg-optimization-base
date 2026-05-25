import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm combining
# random sampling, local coordinate-wise improvement, and a lightweight
# adaptive evolutionary strategy. It continuously tracks the best point found.
# Search state: Maintains current best solution (best_x, best_y), plus a
# "center" point and a step size (sigma) for generating candidate points.
# Candidate generation: At each iteration, generates multiple proposals by
# adding Gaussian noise around a center and also includes a coordinate-wise
# local search (try +/- small steps along each dimension).
# Selection and replacement: Uses greedy selection: if a candidate yields a
# lower objective value than the current best, it replaces the best and the
# center. For the center update, it follows the best improving candidate
# found in the current iteration.
# Adaptation: Step size sigma is reduced when improvements are found and
# expanded when no improvement is observed, based on an improvement counter.
# Exploration mechanisms: Global exploration via Gaussian sampling with a
# sigma proportional to the search range.
# Exploitation mechanisms: Local coordinate-wise perturbations around the
# current best/center to refine along individual dimensions.
# Boundary handling: Candidate points are clipped to the provided bounds.
# Budget strategy: Uses an explicit evaluation counter; the total number of
# objective calls never exceeds the given budget. It stops early if the budget
# is reached, and final returns are the best-so-far.
# Closest known influences: Inspired by simple evolution strategies and
# coordinate descent hybrids, with adaptive step-size behavior.
# Novelty or unusual aspects: Uses a small mix of global and local moves per
# iteration and adapts sigma using whether any move improved the best during
# the iteration.
# Failure modes: In very high dimensions or extremely noisy objectives,
# improvements may be rare, causing sigma to inflate and slow convergence.
# If bounds are degenerate (lb==ub), all points collapse to that value and
# the method relies on a single evaluation; this is handled naturally.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        budget = max(0, self.budget)
        if budget == 0:
            # No evaluations permitted; return a valid point in bounds if possible.
            lb, ub = self._get_bounds(func, d)
            x0 = np.clip((lb + ub) / 2.0, lb, ub)
            return x0, np.inf

        lb, ub = self._get_bounds(func, d)
        span = ub - lb
        # Avoid zero division: if span is zero in a dimension, sigma becomes 0 there.
        # Use a conservative initial sigma relative to the range.
        sigma = 0.3 * np.where(span > 0, span, 1.0)
        sigma = np.clip(sigma, 0.0, np.max(span) if np.max(span) > 0 else 1.0)

        evals = 0

        def clip(x):
            return np.minimum(np.maximum(x, lb), ub)

        # Pick an initial center: midpoint; if degenerate bounds, it's constant anyway.
        center = clip((lb + ub) / 2.0)

        def eval_obj(x):
            nonlocal evals
            if evals >= budget:
                return np.inf
            y = func(x)
            evals += 1
            return float(y)

        best_x = center.copy()
        best_y = eval_obj(best_x)

        # If bounds collapse to a constant or budget is 1, we're done.
        if budget == 1:
            return best_x, best_y

        # Determine per-iteration evaluation allotment.
        # Keep small to limit the chance of exhausting budget too quickly.
        # Roughly aim for 10-50 iterations depending on budget and dimension.
        max_iters = max(1, min(60, budget))
        # Candidate count per iteration: global proposals + local coordinate tries.
        # Ensure at least some exploration.
        global_per_iter = max(2, min(8, budget // max_iters if budget // max_iters > 0 else 2))

        # Local coordinate move step: small fraction of span.
        base_local_step = 0.05 * span
        # If span is zero in a dimension, local step is zero there.

        no_improve_streak = 0

        # Main loop
        for _ in range(max_iters):
            if evals >= budget:
                break

            # ---- Global exploration: Gaussian proposals around center ----
            improved = False
            center_for_gen = center.copy()

            # Generate global candidates, including the center itself (already evaluated),
            # but we only evaluate if remaining budget allows.
            for _k in range(global_per_iter):
                if evals >= budget:
                    break
                # Isotropic-ish Gaussian scaled by sigma per dimension.
                noise = np.random.randn(d) * sigma
                cand = clip(center_for_gen + noise)

                # Avoid reevaluating exact center if unchanged and already best; still ok if bounds collapse.
                y = eval_obj(cand)
                if y < best_y:
                    best_y = y
                    best_x = cand.copy()
                    center = best_x.copy()
                    improved = True

            # ---- Exploitation: coordinate-wise local search around current center ----
            # Try a small subset of coordinates if dim is large to stay within budget.
            if evals < budget:
                remaining = budget - evals
                # Choose up to one move per coordinate, but cap count by remaining and budget usage.
                # Start from best direction by shuffling coordinates to diversify.
                if d <= 24:
                    coord_count = d
                else:
                    coord_count = min(d, 16 + (budget // max(1, d)) * 2)
                coord_count = max(1, min(coord_count, remaining))

                coords = np.random.permutation(d)[:coord_count]
                # Current local step magnitude: shrink with sigma to focus as search progresses.
                local_step = 0.5 * base_local_step + 0.5 * (0.2 * sigma)
                # Ensure not NaN and proper shape.
                local_step = np.where(np.isfinite(local_step), local_step, 0.0)

                for i in coords:
                    if evals >= budget:
                        break
                    step = local_step[i]
                    if step == 0.0:
                        continue
                    # Try +step and -step
                    for sgn in (1.0, -1.0):
                        if evals >= budget:
                            break
                        cand = center.copy()
                        cand[i] = cand[i] + sgn * step
                        cand = clip(cand)
                        y = eval_obj(cand)
                        if y < best_y:
                            best_y = y
                            best_x = cand.copy()
                            center = best_x.copy()
                            improved = True

            # ---- Adaptation: adjust sigma based on whether any improvement happened ----
            if improved:
                no_improve_streak = 0
                # If improving, shrink step size to exploit while keeping some exploration.
                # Also slightly recenters towards best.
                sigma = 0.75 * sigma
                # Prevent sigma from going to exactly zero unless bounds span is zero.
                sigma = np.where(span > 0, np.maximum(sigma, 1e-12 * span), 0.0)
            else:
                no_improve_streak += 1
                # If no improvement, expand slightly to escape local traps, but cap it.
                # Cap sigma to a reasonable fraction of span.
                max_sigma = np.where(span > 0, 0.8 * span, 0.0)
                sigma = np.minimum(1.15 * sigma, max_sigma)
                # Occasional mild random reset of center to re-explore if stuck.
                if no_improve_streak >= 4 and evals < budget:
                    # Jump center towards the current best plus a small random perturbation.
                    jump = np.random.randn(d) * (0.25 * sigma + 1e-12)
                    center = clip(best_x + jump)
                    no_improve_streak = 0

        return best_x, best_y

    @staticmethod
    def _get_bounds(func, dim):
        # Supported conventions:
        # - func.lower / func.upper
        # - func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)
        else:
            raise AttributeError(
                "Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )

        if lb.shape == ():  # scalar
            lb = np.full(dim, float(lb))
        if ub.shape == ():  # scalar
            ub = np.full(dim, float(ub))

        lb = lb.reshape(-1)[:dim]
        ub = ub.reshape(-1)[:dim]

        if lb.shape[0] != dim or ub.shape[0] != dim:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Ensure lb <= ub (if swapped, correct it).
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        return lb2, ub2
