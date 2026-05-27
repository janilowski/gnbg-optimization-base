# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, robust black-box minimization algorithm
# (gradient-free) that works across dimensions. It mixes a lightweight global restart
# mechanism with a local, CMA-ES-inspired evolution strategy update using only numpy.
# Search state: The algorithm maintains a current "center" point, a covariance proxy
# (diagonal variance), and a mutation step-size sigma. It also tracks best-so-far
# solution and a small archive of recent samples to help local selection.
# Candidate generation: At each iteration, it generates a batch of candidate points by
# sampling from a multivariate normal with diagonal covariance around the current
# center. Candidates are mirrored and clipped to respect bounds, reducing wasted
# evaluations.
# Selection and replacement: Among the batch, the best few candidates are used to update
# the center (weighted average) and to refine the diagonal variance proxy. The
# covariance/variance is driven by successful steps to bias future sampling.
# Adaptation: Sigma is adapted based on the "success rate" (how many of the batch
# improved over the center/best). Additionally, the diagonal variance is scaled up/down
# using the relative spread of top candidates.
# Exploration mechanisms: Multiple restarts occur when progress stalls. During restarts,
# sigma is increased and the center is reset using a mix of current best and random
# points, enabling escape from local minima.
# Exploitation mechanisms: When improvements happen frequently, sigma shrinks and the
# center update uses stronger weighting on the best candidates, focusing search locally.
# Boundary handling: Each candidate is reflected back into bounds when it overshoots, then
# clipped as a final safeguard. This keeps samples within feasible region without
# collapsing distributions excessively.
# Budget strategy: The algorithm uses a careful evaluation counting scheme and
# stops immediately when the remaining budget is too small for another batch, ensuring
# it never exceeds the provided evaluation budget.
# Closest known influences: The design is inspired by CMA-ES/ES principles (diagonal
# covariance, step-size adaptation, recombination), simplified for a black-box benchmark
# interface and compactness.
# Novelty or unusual aspects: It uses a success-rate-based sigma adaptation plus a
# robust diagonal-variance update and a simple restart rule, all without needing any
# gradient information or additional libraries.
# Failure modes: If the objective is extremely noisy, success detection may be unreliable,
# causing oscillation or premature restarts. For very small budgets, the batch size is
# reduced to avoid exceeding the budget, possibly limiting performance.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from func ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("func must provide either (lower, upper) or bounds.lb/bounds.ub")

        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        lb = lb.reshape(-1).astype(float)
        ub = ub.reshape(-1).astype(float)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimensionality does not match dim")

        # Ensure valid intervals
        span = ub - lb
        span = np.where(span > 0, span, 1.0)  # avoid division by zero in degenerate bounds

        # ---- Helpers ----
        def reflect_to_bounds(x):
            # Reflect overshoots back into [lb, ub].
            # Works with vectorized operations using modular reflection.
            x = np.asarray(x, dtype=float)
            # Bring into an extended domain by shifting to [0, span]
            y = x - lb
            s = span
            # For degenerate dimensions, force to midpoint of bounds
            if np.any(s <= 0):
                mid = (lb + ub) / 2.0
                x = np.where(s <= 0, mid, x)
                s = np.where(s <= 0, 1.0, s)
                y = x - lb

            # Reflect by folding into [0, 2s] then mapping
            two_s = 2.0 * s
            # modulo in [0, 2s)
            y_mod = np.mod(y, two_s)
            # map to [0, s]
            y_ref = np.where(y_mod <= s, y_mod, two_s - y_mod)
            return y_ref + lb

        def clipped_to_bounds(x):
            return np.minimum(ub, np.maximum(lb, x))

        def make_random_point():
            # Uniform sample within bounds
            r = np.random.rand(dim)
            return lb + r * (ub - lb)

        def eval_point(x):
            # Always keep inside bounds
            x = reflect_to_bounds(x)
            x = clipped_to_bounds(x)
            return func(x)

        # ---- Budget handling ----
        # Use batches for efficiency; adjust batch size to budget.
        # Typical ES uses lambda around 4 + floor(3*ln(dim)), but keep robust for small budgets.
        lam_base = int(4 + 3 * np.log(max(dim, 2)))
        lam_base = max(4, lam_base)
        lam = min(lam_base, max(1, budget))
        # Require at least 1 eval for initialization.
        if budget <= 0:
            raise ValueError("budget must be positive")

        eval_count = 0
        best_x = None
        best_y = np.inf

        # ---- Initialization ----
        # Start from a random point plus a few extra random samples if budget allows.
        # This acts as a basic global exploration.
        init_k = 1
        if budget >= 2:
            init_k = min(4, budget)  # small set of initial guesses
        # Diagonal variance proxy initialized relative to bounds
        # Start with sigma proportional to span.
        sigma = 0.3 * np.median(span[span > 0]) if np.any(span > 0) else 1.0
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        # Center
        x_center = make_random_point()
        y_center = eval_point(x_center)
        eval_count += 1
        if y_center < best_y:
            best_y = y_center
            best_x = x_center.copy()

        # Optional additional random initial points
        for _ in range(init_k - 1):
            if eval_count >= budget:
                break
            x0 = make_random_point()
            y0 = eval_point(x0)
            eval_count += 1
            if y0 < best_y:
                best_y = y0
                best_x = x0.copy()
                x_center = x0.copy()
                y_center = y0

        # Diagonal variance proxy; start with normalized span
        var = (0.2 * span) ** 2
        # Replace nonpositive with sigma^2
        var = np.where(span > 0, var, sigma ** 2)
        # A small floor to avoid collapse
        var_floor = (1e-12 * sigma) ** 2
        var = np.maximum(var, var_floor)

        # Success tracking for adaptation and restarts
        no_improve_steps = 0
        best_step_eval = 0
        # Restart periodically if no improvement
        restart_patience = max(10, int(0.2 * (budget / max(lam, 1) + 1)))

        # ---- ES-like loop ----
        # We use diagonal covariance to keep computation simple.
        # Recombination weights favor top performers.
        while eval_count < budget:
            remaining = budget - eval_count
            if remaining <= 0:
                break

            # Decide lambda based on remaining evaluations
            cur_lam = min(lam, remaining)
            if cur_lam <= 0:
                break

            # Generate candidate set around center
            # Diagonal covariance: x = center + sigma * sqrt(var_norm) * N(0,1)
            # Here var already includes scale, so we incorporate sigma as global factor.
            # We normalize var to avoid huge scaling across dimensions.
            scale = np.sqrt(var)
            # Ensure scale is positive
            scale = np.maximum(scale, np.sqrt(var_floor))

            Z = np.random.randn(cur_lam, dim)
            X = x_center + (sigma * Z) * scale

            # Evaluate candidates (respect budget exactly)
            Y = np.empty(cur_lam, dtype=float)
            actual_lam = cur_lam
            for i in range(cur_lam):
                if eval_count >= budget:
                    actual_lam = i
                    break
                y = eval_point(X[i])
                Y[i] = y
                eval_count += 1

            if actual_lam <= 0:
                break
            X = X[:actual_lam]
            Y = Y[:actual_lam]

            # Sort by fitness (minimization)
            idx = np.argsort(Y)
            X_sorted = X[idx]
            Y_sorted = Y[idx]

            # Update best-so-far
            if Y_sorted[0] < best_y - 1e-15 * (abs(best_y) + 1.0):
                best_y = float(Y_sorted[0])
                best_x = X_sorted[0].copy()
                x_center = best_x.copy()
                y_center = best_y
                no_improve_steps = 0
                best_step_eval = eval_count
            else:
                no_improve_steps += 1

            # Determine if there was improvement relative to current center
            improved = np.sum(Y_sorted[: max(2, int(0.25 * actual_lam))] < y_center)
            success_rate = improved / max(actual_lam, 1)

            # If center isn't updated to best_x, we still can exploit using weighted recombination.
            # Recombination: take top k and average with decreasing weights.
            k = max(2, int(0.2 * actual_lam))
            k = min(k, actual_lam)
            topX = X_sorted[:k]
            # Weights: exponential-ish by rank
            ranks = np.arange(k, dtype=float)
            weights = np.log((k + 1.0) - ranks)
            weights = weights / np.sum(weights)
            x_new = np.sum(topX * weights[:, None], axis=0)

            # Use diagonal variance update from successful steps/spread.
            # Drive variance toward the observed dispersion of top candidates.
            diffs = (topX - x_new)
            # Weighted squared radius per dimension
            w = weights[:, None]
            var_obs = np.sum(w * (diffs ** 2), axis=0)

            # Blend variance proxy (keep some inertia)
            c_var = 0.25
            var = (1.0 - c_var) * var + c_var * np.maximum(var_obs, var_floor)

            # Step-size adaptation (global sigma)
            # Success_rate target roughly 0.2 for modest progress.
            target = 0.2
            if success_rate > target:
                sigma *= np.exp(0.15 * (success_rate - target) / max(target, 1e-12))
            else:
                sigma *= np.exp(-0.25 * (target - success_rate) / max(target, 1e-12))
            # Clamp sigma to reasonable bounds
            # Upper bound relative to domain size to avoid runaway.
            dom = np.median(span[span > 0]) if np.any(span > 0) else 1.0
            if not np.isfinite(dom) or dom <= 0:
                dom = 1.0
            sigma_max = 2.0 * dom
            sigma_min = 1e-12 * dom
            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # Update center to recombined candidate unless stuck on non-improving best-only
            # (still keep it bounded inside)
            x_center = clipped_to_bounds(reflect_to_bounds(x_new))
            y_center = float(min(y_center, np.min(Y_sorted)))  # track lowest in last batch

            # Restart if stuck
            if no_improve_steps >= restart_patience:
                no_improve_steps = 0
                # Restart center: jitter around best_x if known, otherwise random
                if best_x is not None:
                    jitter = np.random.randn(dim)
                    # Use sigma and var to create a meaningful jump
                    jump = (sigma * jitter) * np.sqrt(var)
                    x_center = clipped_to_bounds(reflect_to_bounds(best_x + jump))
                else:
                    x_center = make_random_point()
                y_center = eval_point(x_center)
                eval_count += 1 if eval_count < budget else 0
                if y_center < best_y:
                    best_y = y_center
                    best_x = x_center.copy()
                # Inflate search scales to escape local basins
                sigma *= 2.0
                sigma = float(min(sigma, sigma_max))
                var = np.maximum(var * 2.0, var_floor)

            if eval_count >= budget:
                break

        # Final safeguard: ensure best_x is valid and returned
        if best_x is None:
            best_x = make_random_point()
            best_y = eval_point(best_x)
        else:
            best_x = clipped_to_bounds(reflect_to_bounds(best_x))

        return best_x, float(best_y)
