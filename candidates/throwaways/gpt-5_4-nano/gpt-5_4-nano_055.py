# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm based on CMA-ES-like
# sampling with diagonal covariance adaptation and occasional multi-start restarts.
# It tracks the best-so-far solution and uses objective evaluations to update
# a simple covariance model.
# Search state: Maintains current mean vector (x_mean), diagonal step-size
# (sigma), and diagonal covariance scales (var) that define an elliptical search
# distribution around the mean. Tracks evaluation count and best (x, y).
# Candidate generation: Each iteration samples a population of offspring by drawing
# z ~ N(0, I) and forming x = clamp(x_mean + sigma * (sqrt(var) * z), bounds).
# Also adds one "reflection-like" candidate around the current best to improve
# robustness to bound effects.
# Selection and replacement: Evaluates all candidates, then selects the top half
# (lowest objective values) to update the mean as a weighted average toward
# better candidates (plus a slight pull toward the best). Best-so-far is kept.
# Adaptation: Updates diagonal variance using weighted contributions from selected
# points (relative to mean), and adapts sigma using a success rule based on
# whether the best of the generation improved over the previous best.
# Exploration mechanisms: Random Gaussian sampling controlled by sigma and var,
# periodic restart when progress stalls to escape local minima, and reflection-like
# candidate sampling to handle boundary plateaus.
# Exploitation mechanisms: Weighted mean update toward low objective candidates,
# sigma reduction after successful progress, and reduced variance growth after poor
# generations.
# Boundary handling: Uses a clamp-to-bounds strategy for each coordinate, and
# downweights variance adaptation when many coordinates are clamped, to avoid
# wasting evaluations on saturated boundaries.
# Budget strategy: Uses a fixed number of evaluations per iteration (population size)
# and ensures it never exceeds the provided evaluation budget; the final partial
# iteration is handled safely.
# Closest known influences: Inspired by CMA-ES (sampling, selection, covariance adaptation)
# and evolutionary strategies (mu+lambda style with diagonal covariance simplification),
# but implemented in a compact, diagonal form.
# Novelty or unusual aspects: Adds a small, bound-aware "best-reflection" candidate
# each generation and uses a diagonal variance with a clamp-aware shrink factor.
# Failure modes: If the objective is extremely noisy or adversarially scaled,
# the simple success rule may oscillate; restarts mitigate but may still waste budget.
# Very low evaluation budgets may limit adaptation; the algorithm defaults to
# conservative initialization from the center of bounds.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # --- Bounds handling ---
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("func must provide either (lower, upper) or bounds.lb/bounds.ub")

        n = self.dim
        if lb.shape[0] != n or ub.shape[0] != n:
            raise ValueError("Bounds dimension does not match dim")

        # Ensure valid bounds
        lb = np.minimum(lb, ub)
        ub = np.maximum(ub, lb)
        width = ub - lb
        width = np.where(width > 0, width, 1.0)  # avoid zero-width issues

        # --- Objective wrapper with budget enforcement ---
        evals = 0
        budget = self.budget

        def eval_obj(x):
            nonlocal evals
            if evals >= budget:
                # Should never happen if we respect remaining evaluations.
                return float("inf")
            evals += 1
            # func may accept list/np array; provide np array
            y = func(x)
            return float(y)

        # --- Initialize search state ---
        # Start from the center of the box with moderate sigma based on width.
        x_mean = (lb + ub) / 2.0
        diag_var = np.ones(n, dtype=float)
        # sigma around 0.25 of typical box width (scaled per coordinate)
        sigma = 0.25 * float(np.mean(width))

        best_x = np.clip(x_mean, lb, ub).copy()
        best_y = eval_obj(best_x)

        # Population sizes:
        # For small budgets, fall back to simpler single-point sampling.
        # For larger budgets, use an ES-like population.
        # lambda ~ 4 + 3*log(n) but cap to reasonable size.
        lam = int(np.clip(4 + 3 * np.log(max(2, n)), 4, 24))
        mu = max(2, lam // 2)

        # Restart logic
        no_improve_iters = 0
        stall_limit = 8  # generations without improvement before restart
        restarts_left = 2  # keep it small to conserve budget

        # Helper: clamp and measure clamp ratio
        def clamp_and_ratio(x):
            x_clamped = np.minimum(np.maximum(x, lb), ub)
            # clamp ratio per candidate: fraction of coordinates clamped
            ratio = np.mean(x_clamped != x)
            return x_clamped, ratio

        prev_best = best_y

        # Main loop: each generation consumes up to lam evaluations.
        while evals < budget:
            remaining = budget - evals
            cur_lam = min(lam, remaining)
            if cur_lam <= 0:
                break

            # --- Generate population ---
            # Offspring: x = x_mean + sigma * (sqrt(var) * z)
            # Use diagonal covariance: var * z^2 directionally scaled.
            std = sigma * np.sqrt(np.maximum(diag_var, 1e-12))
            Z = np.random.randn(cur_lam, n)
            X = x_mean[None, :] + Z * std[None, :]

            # Clamp candidates
            Xc = np.empty_like(X)
            clamp_ratios = np.empty(cur_lam, dtype=float)
            for i in range(cur_lam):
                xi, cr = clamp_and_ratio(X[i])
                Xc[i] = xi
                clamp_ratios[i] = cr

            # Add reflection-like candidate around best (if there's room)
            # This is done by replacing the worst candidate in the batch later
            # to avoid changing evaluation count.
            # We'll evaluate it unconditionally by overwriting one candidate.
            if cur_lam >= 2:
                # Construct a reflection around best_x and mean (bound-aware):
                # x_ref = best_x + 0.5*(best_x - x_mean)
                x_ref = best_x + 0.5 * (best_x - x_mean)
                x_ref = np.minimum(np.maximum(x_ref, lb), ub)
                # Put it in last slot to ensure we still evaluate exactly cur_lam
                Xc[-1] = x_ref
                clamp_ratios[-1] = np.mean(x_ref != (best_x + 0.5 * (best_x - x_mean)))

            # --- Evaluate ---
            Y = np.empty(cur_lam, dtype=float)
            for i in range(cur_lam):
                Y[i] = eval_obj(Xc[i])

            # Update best-so-far
            gen_best_idx = int(np.argmin(Y))
            gen_best_y = Y[gen_best_idx]
            gen_best_x = Xc[gen_best_idx]

            if gen_best_y < best_y:
                best_y = gen_best_y
                best_x = gen_best_x.copy()

            improved = gen_best_y < prev_best
            if improved:
                no_improve_iters = 0
            else:
                no_improve_iters += 1
            prev_best = min(prev_best, gen_best_y)

            # --- Select and update mean ---
            # Sort by objective (minimization)
            order = np.argsort(Y)
            sel_idx = order[:mu]
            sel_X = Xc[sel_idx]
            sel_Y = Y[sel_idx]

            # Weighting: stronger weight for better points
            # Use rank-based weights to reduce sensitivity to noise scaling.
            ranks = np.arange(mu, dtype=float)
            # Higher rank (better) => larger weight:
            # order: 0 best ... mu-1 worst within selection
            w = (mu - ranks)
            w = w / np.sum(w)
            w = w.astype(float)

            # Compute new mean toward selected points
            x_new = np.sum(sel_X * w[:, None], axis=0)

            # Slight exploitation pull toward global best
            # Helps when selection diversity is limited.
            pull = 0.15
            x_new = (1.0 - pull) * x_new + pull * best_x

            # --- Diagonal variance update (CMA-ES-like) ---
            # Update var based on selected deviations from current mean.
            # var <- (1-c)*var + c*(weighted dev^2 / sigma^2)
            # Incorporate clamp information: if many candidates hit bounds, shrink var.
            dev = (sel_X - x_mean[None, :]) / max(sigma, 1e-12)
            # Weighted average of squared deviations per coordinate
            var_target = np.sum((dev ** 2) * w[:, None], axis=0)

            clamp_factor = 1.0 - np.clip(np.mean(clamp_ratios), 0.0, 1.0) * 0.5
            var_target *= clamp_factor

            c_var = 0.25  # adaptation rate for diagonal covariance
            diag_var = (1.0 - c_var) * diag_var + c_var * np.maximum(var_target, 1e-10)

            # Keep var from exploding
            diag_var = np.clip(diag_var, 1e-8, 1e3)

            # Update sigma with success rule
            # If improving, reduce sigma slightly (exploit); otherwise increase a bit (explore).
            if improved:
                sigma *= 0.92
            else:
                sigma *= 1.06

            # Ensure sigma stays within reasonable scale relative to bounds width
            sigma_min = 1e-12
            sigma_max = 0.9 * float(np.mean(width))
            sigma = float(np.clip(sigma, sigma_min, sigma_max))

            # Update mean
            x_mean = np.minimum(np.maximum(x_new, lb), ub)

            # --- Restart handling when stalled ---
            if no_improve_iters >= stall_limit and restarts_left > 0 and evals < budget:
                restarts_left -= 1
                no_improve_iters = 0
                # Recenter around best and increase exploration
                x_mean = best_x.copy()
                diag_var = np.ones(n, dtype=float)
                sigma *= 1.6

        return best_x, best_y
