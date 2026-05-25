import numpy as np


# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, budgeted black-box minimization strategy
# (derivative-free) using an adaptive population of candidate points. It mixes
# global exploration via random directions and local exploitation via a
# coordinate-wise search around the current best.
# Search state: Tracks best solution (x_best, y_best), a current step size
# (sigma) for perturbations, and a small population that is repeatedly
# regenerated each iteration. Maintains a counter of objective evaluations to
# never exceed the provided evaluation budget.
# Candidate generation: Each iteration samples candidates around the best
# using either Gaussian perturbations scaled by sigma or random directions
# with reflection-style boundary handling. It also adds occasional uniform
# samples for diversity.
# Selection and replacement: Evaluates all candidates in the generated batch,
# then selects the lowest objective value as the new best. The entire
# population is replaced each iteration (plus elitism of the current best).
# Adaptation: If the best improves in an iteration, sigma is gently increased
# (or kept larger) to exploit; if not, sigma is decreased to refine locally.
# Exploration mechanisms: Diversity via random uniform points and broader
# perturbations when improvements are absent.
# Exploitation mechanisms: When improvements occur, the algorithm focuses by
# shrinking sigma and using a local coordinate sweep (small positive/negative
# moves in each dimension) around the best.
# Boundary handling: Uses reflection to keep points within bounds. If
# bounds are degenerate or missing, it falls back to a conservative default
# [-1, 1] interval per dimension.
# Budget strategy: Splits the budget into batches, ensuring all evaluations
# are accounted for and that no call to the objective exceeds the remaining
# evaluations.
# Closest known influences: Heuristic resembles CMA-ES-like step adaptation
# (simple sigma control) combined with a coordinate-wise local search, but
# deliberately avoids covariance estimation for compactness.
# Novelty or unusual aspects: A hybrid "batch evolutionary sampling + optional
# coordinate sweep" with reflection boundary handling, designed to be robust
# across varying dimensions while remaining budget-safe.
# Failure modes: If the objective is extremely noisy or highly deceptive,
# sigma adaptation may oscillate. Very tight bounds or wrong bound inference
# could reduce performance. In worst cases, it may behave like random search
# if improvements are rarely observed.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        n_eval = 0
        dim = self.dim

        # --- Bounds extraction ---
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = getattr(func, "lower")
            ub = getattr(func, "upper")
        elif hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = getattr(b, "lb")
                ub = getattr(b, "ub")

        if lb is None or ub is None:
            # Conservative fallback.
            lb = -np.ones(dim, dtype=float)
            ub = np.ones(dim, dtype=float)
        else:
            lb = np.asarray(lb, dtype=float).reshape(-1)
            ub = np.asarray(ub, dtype=float).reshape(-1)
            if lb.size != dim:
                # If bounds are scalar-like, broadcast; otherwise clamp length.
                if lb.size == 1:
                    lb = np.full(dim, float(lb[0]))
                else:
                    lb = np.resize(lb, dim)
            if ub.size != dim:
                if ub.size == 1:
                    ub = np.full(dim, float(ub[0]))
                else:
                    ub = np.resize(ub, dim)

        # Ensure valid ordering.
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        width = hi - lo
        # Avoid zero width divisions; keep non-negative.
        width = np.maximum(width, 0.0)

        def reflect_to_bounds(x):
            # Reflect x into [lo, hi] for each coordinate, elementwise.
            # For numerical robustness, handle near-degenerate intervals.
            x = np.asarray(x, dtype=float)
            if np.all(width <= 0):
                return np.clip(x, lo, hi)  # all constant bounds
            y = x.copy()
            for i in range(dim):
                if width[i] <= 0:
                    y[i] = lo[i]
                    continue
                # Map to [0, 2W) then reflect to [0, W]
                w = width[i]
                # Translate to 0.. then apply modulo.
                t = (y[i] - lo[i]) % (2.0 * w)
                if t < w:
                    y[i] = lo[i] + t
                else:
                    y[i] = lo[i] + (2.0 * w - t)
            return y

        def eval_objective(x):
            nonlocal n_eval
            if n_eval >= self.budget:
                # Should not happen due to careful budget checks.
                return np.inf
            y = func(x)
            n_eval += 1
            return float(y)

        # If budget is extremely small, just sample within bounds.
        if self.budget <= 0:
            # No evaluation possible; define a deterministic output.
            x0 = lo.copy()
            return x0, np.inf

        # --- Hyperparameters (kept compact & adaptive) ---
        # Initial sigma based on the scale of bounds.
        # If width is all zero, keep sigma small.
        sigma = 0.2 * np.sqrt(np.mean((width + 1e-12) ** 2))
        if sigma <= 0 or not np.isfinite(sigma):
            sigma = 1.0

        # Batch size: start moderate, adjust to remaining budget.
        # Use population size that works well in low/high dims.
        base_batch = 4 + int(np.sqrt(dim))
        batch_size = max(2, min(base_batch, self.budget))

        # Diversity probability (uniform exploration).
        p_uniform = min(0.35, 2.0 / (1.0 + dim ** 0.5))
        improve_streak = 0

        # --- Initial candidate ---
        # Start from a random point and optionally a center point.
        # Always evaluate at least one point.
        x_best = lo + (hi - lo) * np.random.rand(dim)
        x_best = reflect_to_bounds(x_best)
        y_best = eval_objective(x_best)

        # Also evaluate mid if budget allows (often helpful).
        if n_eval < self.budget and self.budget >= 2:
            x_mid = 0.5 * (lo + hi)
            x_mid = reflect_to_bounds(x_mid)
            y_mid = eval_objective(x_mid)
            if y_mid < y_best:
                x_best, y_best = x_mid, y_mid

        it = 0
        while n_eval < self.budget:
            it += 1
            remaining = self.budget - n_eval
            # Ensure we evaluate at most remaining evaluations.
            current_batch = min(batch_size, remaining)

            # Generate candidates, including elitism.
            # We'll keep x_best as one candidate to ensure feasibility.
            candidates = []
            candidates.append(x_best.copy())

            # Determine how strongly to sample.
            # If no recent improvements, temporarily increase exploration.
            if improve_streak <= 0:
                sigma_scale = 1.25
            else:
                sigma_scale = 0.95

            # Candidate generation: mixture of Gaussian around best and occasional uniform.
            while len(candidates) < current_batch:
                if np.random.rand() < p_uniform:
                    x = lo + (hi - lo) * np.random.rand(dim)
                else:
                    # Use Gaussian perturbation; occasionally use random directions.
                    if np.random.rand() < 0.3:
                        # Random direction scaled by sigma.
                        d = np.random.randn(dim)
                        nd = np.linalg.norm(d) + 1e-12
                        d = d / nd
                        # Bias step length a bit towards smaller improvements early.
                        step = np.random.rand() * (sigma * sigma_scale)
                        x = x_best + step * d
                    else:
                        x = x_best + (sigma * sigma_scale) * np.random.randn(dim)

                x = reflect_to_bounds(x)
                candidates.append(x)

            # Evaluate candidates and select best.
            y_local_best = y_best
            x_local_best = x_best
            for x in candidates:
                if n_eval >= self.budget:
                    break
                y = eval_objective(x)
                if y < y_local_best:
                    y_local_best = y
                    x_local_best = x.copy()

            # Adapt sigma based on improvement.
            if y_local_best < y_best - 1e-15:
                improve_streak += 1
                x_best = x_local_best
                y_best = y_local_best
                # Exploitation: shrink less aggressively; allow sigma to recover slightly.
                sigma = sigma * (0.92 ** (-1))  # increase a bit
                sigma = min(sigma, 0.5 * (np.max(width) + 1.0) + 1e-12)
            else:
                improve_streak = 0
                # Exploration/exploitation switch: shrink to refine.
                sigma = sigma * 0.85

            # Optional coordinate-wise local search around the best.
            # Do this only occasionally and only if budget allows.
            remaining = self.budget - n_eval
            if remaining > 2 * dim and (it % 3 == 0):
                # Small step relative to sigma and bounds.
                # Use per-coordinate step sizes based on width (if width exists).
                step_base = max(1e-12, sigma * 0.35)
                # Coordinate sweep: test +/- along each axis until budget is near limit.
                for j in range(dim):
                    if self.budget - n_eval < 2:
                        break
                    step_j = step_base
                    if width[j] > 0:
                        step_j = min(step_j, 0.25 * width[j] + 1e-12)

                    # Positive move
                    x_p = x_best.copy()
                    x_p[j] = x_p[j] + step_j
                    x_p = reflect_to_bounds(x_p)
                    y_p = eval_objective(x_p)

                    if y_p < y_best - 1e-15:
                        x_best, y_best = x_p, y_p
                        continue

                    # Negative move
                    x_m = x_best.copy()
                    x_m[j] = x_m[j] - step_j
                    x_m = reflect_to_bounds(x_m)
                    y_m = eval_objective(x_m)

                    if y_m < y_best - 1e-15:
                        x_best, y_best = x_m, y_m

                # After local search, slight sigma tightening.
                sigma = sigma * 0.9

            # Recompute batch size occasionally to react to dimension changes in budget-limited runs.
            # Keep it bounded to avoid overshooting remaining budget.
            batch_size = max(2, min(8 + int(np.sqrt(dim)), self.budget - n_eval if self.budget - n_eval > 1 else 2))

            # If sigma becomes extremely small and no improvement occurs, use uniform exploration more.
            if sigma < 1e-12:
                sigma = 0.1 * (np.max(width) + 1.0)
                p_uniform = min(0.6, p_uniform + 0.05)

        return x_best, y_best
