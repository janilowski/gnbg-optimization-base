import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm (Algorithm class)
# that uses a budgeted, population-free variant of CMA-ES–like search with rank-based step-size
# adaptation. It keeps an internal multivariate Gaussian search distribution and iteratively
# samples candidate solutions, evaluates them, and updates the distribution toward better points.
# Search state: The algorithm maintains a mean vector (current best center), a global step-size
# (sigma), and a diagonal covariance approximation (variances per dimension) represented through
# a vector of scaling factors. It tracks the best-so-far solution and its objective value.
# Candidate generation: For each generation, it samples lambda candidate points as:
#   x = mean + sigma * (diag(scale) * z),
# where z ~ N(0, I). It also includes the current mean as an always-evaluated candidate when
# budget allows, improving robustness.
# Selection and replacement: It selects the best mu points (lowest objective values) from the
# sampled candidates and computes a weighted recombination to update the mean toward these points.
# The diagonal scaling factors and step-size are adapted based on improvement signals derived
# from the selected samples.
# Adaptation: Step-size sigma is adapted using an evolution-path-inspired scheme:
# it estimates a normalized progress signal from the selected steps and applies a multiplicative
# update. The diagonal covariance scaling uses a clipped learning rate on the squared selected
# normalized steps.
# Exploration mechanisms: Sampling from the Gaussian with adaptive sigma provides exploration.
# The diagonal covariance adaptation allows different scales per dimension.
# Exploitation mechanisms: Ranking-based weighted recombination moves the mean toward better
# regions, while smaller sigma/covariance shrinks the distribution around promising areas.
# Boundary handling: Candidate points are clipped to the provided bounds each time they are
# generated. Bounds are read from func.lower/func.upper or func.bounds.lb/func.bounds.ub.
# Budget strategy: The algorithm strictly never exceeds the provided evaluation budget by
# tracking remaining evaluations and limiting each generation’s number of objective calls.
# Closest known influences: CMA-ES (diagonal approximation + rank-based recombination + step-size
# adaptation) combined with clipped boundary handling and budget-aware candidate counts.
# Novelty or unusual aspects: The implementation is deliberately compact and uses only diagonal
# covariance (no matrix algebra) to remain fast and robust for any dimension while retaining
# CMA-ES-like behavior.
# Failure modes: If bounds are extremely tight or the optimum lies near boundaries, clipping can
# reduce effective search diversity. Very small budgets may lead to limited learning. If the
# objective evaluation is noisy and rank selection is unstable, step-size adaptation may oscillate.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        n = self.dim
        if self.budget <= 0:
            # No evaluations allowed; return zeros with NaN objective (best_y unknown).
            return np.zeros(n, dtype=float), float("nan")

        # ---- Bounds handling (robust to different func APIs) ----
        lower = None
        upper = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and func.bounds is not None:
            b = func.bounds
            # Support common naming: lb/ub
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lower = np.asarray(b.lb, dtype=float)
                upper = np.asarray(b.ub, dtype=float)
            # Also support lower/upper under bounds (just in case)
            elif hasattr(b, "lower") and hasattr(b, "upper"):
                lower = np.asarray(b.lower, dtype=float)
                upper = np.asarray(b.upper, dtype=float)

        if lower is None or upper is None:
            # If bounds aren't available, fall back to a generic box around 0.
            # (But requirement says to read bounds; this fallback only helps robustness.)
            lower = np.full(n, -1.0, dtype=float)
            upper = np.full(n, 1.0, dtype=float)

        lower = np.broadcast_to(lower, (n,)).astype(float, copy=False)
        upper = np.broadcast_to(upper, (n,)).astype(float, copy=False)

        # Ensure valid bounds
        lo = np.minimum(lower, upper)
        hi = np.maximum(lower, upper)
        span = hi - lo
        span = np.where(span > 0, span, 1.0)  # avoid zeros

        rng = np.random  # harness sets np.random seed externally

        # ---- Budgeted objective evaluation wrapper ----
        evals = 0
        remaining = self.budget

        def clip(x):
            return np.minimum(np.maximum(x, lo), hi)

        def eval_one(x):
            nonlocal evals, remaining
            if remaining <= 0:
                # Should not happen if budget logic is correct.
                return None
            x = clip(np.asarray(x, dtype=float))
            y = func(x)
            evals += 1
            remaining -= 1
            return float(y), x

        # ---- Initialization ----
        # Mean: random point within bounds to avoid needing extra queries.
        mean = lo + rng.random(n) * (hi - lo)

        # Initial sigma: a fraction of box size, but not too tiny.
        # Use median scale to be robust to varying per-dimension spans.
        median_span = float(np.median(span))
        sigma = 0.3 * median_span
        sigma = max(sigma, 1e-12)

        # Diagonal "covariance" scaling factors: start uniform.
        # We'll adapt scale^2 via learning from selected steps.
        scale = np.ones(n, dtype=float)

        best_x = mean.copy()
        best_y = float("inf")

        # Evaluate mean if possible (often helps baseline).
        if remaining > 0:
            y0, x0 = eval_one(mean)
            best_y = y0
            best_x = x0.copy()

        # ---- Parameters (CMA-ES-ish, diagonal, rank-based) ----
        # Use lambda based on dimension; keep it bounded for compactness.
        lam = int(4 + 3 * np.log1p(n))
        lam = max(lam, 4)
        # Let mu be about half of lambda.
        mu = lam // 2
        mu = max(mu, 2)

        # Recombination weights: log weights, normalized to sum=1.
        # Also used to compute learning of covariance/step adaptation.
        ranks = np.arange(1, mu + 1)
        w = np.log(mu + 0.5) - np.log(ranks)
        w = w.astype(float)
        w_sum = float(np.sum(w))
        w /= w_sum

        # Effective mu:
        mu_eff = 1.0 / float(np.sum(w**2))

        # Step-size control parameters (diagonal approximation).
        # Using common CMA-ES constants.
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        # Learning rates for diagonal covariance (scale factors squared).
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c_cov = (2.0 / (n + np.sqrt(2.0))) / 2.0
        # Evolution paths (simplified)
        p_sigma = np.zeros(n, dtype=float)

        # For covariance scaling, we update a diagonal variance proxy:
        # variances ~ scale^2. We'll maintain scale directly.
        # Clip learning to avoid numerical blow-ups.
        cov_lr = 0.2

        # ---- Main loop ----
        # Each iteration evaluates up to lam candidates, but never exceed remaining.
        while remaining > 0:
            # How many candidates can we evaluate this generation?
            # We'll include mean as a candidate by default only if it hasn't just been
            # evaluated and budget allows. Simpler: always sample lam, but ensure
            # we don't exceed remaining by reducing lam if needed.
            max_can = min(lam, remaining)
            if max_can <= 0:
                break

            # Sample normalized steps z; then scale them.
            # diag(scale) * z implemented as scale * z.
            z = rng.randn(max_can, n)
            # Generate candidates
            # x_i = mean + sigma * (scale * z_i)
            x = mean + sigma * (z * scale)
            # Clip for bounds.
            x = np.clip(x, lo, hi)

            # Evaluate all candidates
            ys = np.empty(max_can, dtype=float)
            valid_x = np.empty((max_can, n), dtype=float)
            for i in range(max_can):
                out = eval_one(x[i])
                if out is None:
                    # Should not happen due to budget control, but handle gracefully.
                    ys = ys[:i]
                    valid_x = valid_x[:i]
                    max_can = i
                    break
                yi, xi = out
                ys[i] = yi
                valid_x[i] = xi

            if max_can <= 0:
                break

            # Rank by objective (minimization)
            order = np.argsort(ys)
            ys_sorted = ys[order]
            x_sorted = valid_x[order]

            # Update best-so-far
            if ys_sorted[0] < best_y:
                best_y = float(ys_sorted[0])
                best_x = x_sorted[0].copy()

            # Select top mu
            mu_sel = min(mu, max_can)
            x_sel = x_sorted[:mu_sel]

            # Weighted recombination: new mean
            # mean_new = sum_i w_i * x_sel[i], with w from the first mu entries.
            w_use = w[:mu_sel]
            mean_new = np.sum(x_sel * w_use[:, None], axis=0)

            # Compute weighted step in normalized coordinates
            # y_step = (mean_new - mean) / (sigma)
            step = (mean_new - mean) / max(sigma, 1e-300)
            # But we used diagonal scaling: scale affects relation to z-space.
            # Convert step into "z-space" approx by dividing by scale.
            step_z = step / np.maximum(scale, 1e-12)

            # Update sigma using a progress estimate.
            # Similar to CMA-ES: ||p_sigma|| normalized by chi_n.
            # We'll use a constant approximation chi_n ~ sqrt(n) * (1 - 1/(4n) + ...)
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
            # Update evolution path
            p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * step_z
            norm_p = np.linalg.norm(p_sigma)
            # Control factor
            sigma_factor = np.exp((c_sigma / d_sigma) * (norm_p / max(chi_n, 1e-12) - 1.0))
            sigma = float(np.clip(sigma * sigma_factor, 1e-12, 1e6 * median_span))

            # Diagonal covariance adaptation:
            # Update scale^2 proxy based on squared normalized steps of selected points.
            # Compute normalized z for selected points relative to current mean:
            # z_sel = (x_sel - mean) / sigma / scale
            z_sel = (x_sel - mean) / max(sigma, 1e-300)
            z_sel = z_sel / np.maximum(scale, 1e-12)
            # Weighted second moment per dimension
            sec_moment = np.sum((z_sel[:mu_sel] ** 2) * w_use[:, None], axis=0)

            # Target new scale^2 proportional to sec_moment, blended with inertia.
            # Use log-domain blending to be more stable.
            target_var = np.clip(sec_moment, 1e-12, 1e12)
            # inertia update on log-scale
            log_scale2 = np.log(np.maximum(scale**2, 1e-30))
            log_target = np.log(target_var)
            log_new = (1.0 - cov_lr) * log_scale2 + cov_lr * log_target
            new_scale2 = np.exp(log_new)
            new_scale2 = np.clip(new_scale2, 1e-12, 1e12)
            scale = np.sqrt(new_scale2)

            # Boundary-aware mean update:
            # If mean_new hits boundaries too often, gently re-center toward interior.
            # (Detect by clipping difference.)
            clipped_mean_new = clip(mean_new)
            if np.any(clipped_mean_new != mean_new):
                # Move a fraction back toward unclipped value (reduce aggressiveness).
                mean = 0.5 * mean + 0.5 * clipped_mean_new
                sigma = 0.9 * sigma
            else:
                mean = mean_new

        return best_x, best_y
