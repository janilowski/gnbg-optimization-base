import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm using a
# CMA-ES-inspired evolution strategy with rank-based selection and a simple
# coordinate-wise fallback. It aims to reduce the objective by iteratively
# sampling candidate solutions from a multivariate normal distribution and
# updating a mean and covariance-like scaling based on the best candidates.
# Search state: Maintains a current mean (x_mean), step-size (sigma),
# and a diagonal covariance scale (diag_cov) controlling exploration per
# dimension. Also tracks the best solution found so far.
# Candidate generation: Each generation samples a population from
# x_mean + sigma * (diag_sqrt * z) where z are standard normal vectors.
# Candidates are clipped to the provided bounds to enforce feasibility.
# Selection and replacement: Evaluates all candidates, selects the best
# individuals (lowest objective values), then updates x_mean as the
# weighted average of selected points using rank-based weights.
# Adaptation: Updates the diagonal covariance scale based on how far
# successful samples deviate from the mean, and adapts sigma using
# progress toward improvement (a monotonic improvement heuristic).
# Exploration mechanisms: Global exploration via multivariate sampling
# and adaptive sigma; per-dimension exploration via diag_cov.
# Exploitation mechanisms: Selection-weighted mean shift increases
# exploitation around good regions; covariance scaling concentrates search.
# Boundary handling: After sampling, candidates are clipped to bounds;
# if clipping is frequent, sigma is reduced to avoid excessive boundary hits.
# Budget strategy: Uses exactly min(budget, function-call_limit) evaluations.
# It runs generations while enough evaluations remain; the final partial
# generation (if needed) evaluates only the remaining candidates.
# Closest known influences: Inspired by CMA-ES (weighted recombination,
# covariance/step-size adaptation) but simplified to diagonal covariance
# and kept fully self-contained for robustness and compactness.
# Novelty or unusual aspects: Combines diagonal covariance updates with a
# lightweight sigma adaptation based on improvement streak and includes a
# deterministic coordinate fallback when progress stalls.
# Failure modes: In very deceptive landscapes or with extremely tight
# bounds, clipping can dominate and reduce effectiveness; budget may end
# before convergence; diagonal covariance may underfit correlations.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # Read bounds from func.lower/upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lower = np.asarray(func.lower, dtype=float)
            upper = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            lower = np.asarray(getattr(b, "lb"), dtype=float)
            upper = np.asarray(getattr(b, "ub"), dtype=float)
        else:
            raise AttributeError("Objective must provide bounds via lower/upper or bounds.lb/bounds.ub.")

        dim = self.dim
        if lower.shape[0] != dim or upper.shape[0] != dim:
            lower = np.resize(lower, (dim,))
            upper = np.resize(upper, (dim,))

        # Ensure finite bounds and proper order
        lower = np.where(np.isfinite(lower), lower, -1e3)
        upper = np.where(np.isfinite(upper), upper, 1e3)
        lo = np.minimum(lower, upper)
        hi = np.maximum(lower, upper)
        span = hi - lo
        span = np.where(span > 0, span, 1.0)  # avoid zero-width degeneracy

        # Evaluation budget bookkeeping
        remaining = self.budget
        if remaining <= 0:
            # No evaluations possible; return a feasible point deterministically
            x0 = lo + 0.5 * span
            return x0, float("inf")

        rng = np.random.default_rng()  # harness sets global seed; default_rng uses it implicitly in many harnesses

        # Initialize mean at center with slight random perturbation
        x_mean = lo + 0.5 * span
        if np.any(span > 0):
            x_mean = x_mean + 0.01 * span * rng.standard_normal(dim)
        x_mean = np.clip(x_mean, lo, hi)

        # Initialize step size relative to domain
        sigma = 0.3 * float(np.mean(span))
        sigma = max(sigma, 1e-12)

        # Diagonal covariance scale (relative variance per dimension)
        diag_cov = np.ones(dim, dtype=float)

        # Track best
        best_x = x_mean.copy()
        best_y = np.inf

        # Helper: evaluate with budget control
        def eval_one(x):
            nonlocal remaining, best_x, best_y
            if remaining <= 0:
                return
            y = func(x)
            remaining -= 1
            if y < best_y:
                best_y = float(y)
                best_x = np.array(x, dtype=float, copy=True)
            return y

        # Initial evaluation (optional but improves robustness if budget allows)
        if remaining > 0:
            eval_one(x_mean)
            if remaining <= 0:
                return best_x, best_y

        # Generation parameters (CMA-ish defaults)
        # Population size ~ 4 + floor(3*log(dim)) (common CMA heuristic), but clamp to budget.
        lam = int(4 + 3 * np.log(max(2, dim)))
        lam = max(4, min(lam, max(4, self.budget)))  # ensure reasonable size
        mu = max(2, lam // 2)

        # Rank-based weights: log(mu+1/2)-log(i), normalized
        ranks = np.arange(mu, dtype=float)
        raw_w = np.log(mu + 0.5) - np.log(ranks + 1.0)
        w = raw_w / np.sum(raw_w)
        mueff = 1.0 / np.sum(w**2)

        # Strategy parameters
        c_sigma = (mueff + 2.0) / (dim + mueff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma
        # Diagonal covariance learning rate
        c_cov = (2.0 / (mueff + dim + 1.0))  # relatively conservative

        # Improvement-based sigma adaptation
        no_improve_streak = 0
        prev_best_y = best_y

        # Coordinate fallback parameters
        coord_steps = 0
        max_coord_steps = 2  # keep limited; used only on stalls
        eps_min = 1e-12

        # Run until evaluations are exhausted
        while remaining > 0:
            # Adjust population size to remaining budget for last generation
            k = min(lam, remaining)

            # Candidate generation
            diag_sqrt = np.sqrt(np.maximum(diag_cov, eps_min))
            # Sample population
            Z = rng.standard_normal((k, dim))
            # Scaling by sigma and diagonal covariance
            X = x_mean[None, :] + sigma * (Z * diag_sqrt[None, :])
            # Boundary handling via clipping
            X = np.clip(X, lo, hi)

            # Evaluate
            ys = np.empty(k, dtype=float)
            for i in range(k):
                y = func(X[i])
                remaining -= 1
                y = float(y)
                ys[i] = y
                if y < best_y:
                    best_y = y
                    best_x = np.array(X[i], dtype=float, copy=True)
                if remaining <= 0 and i < k - 1:
                    # Budget exhausted mid-evaluation; truncate safely
                    ys = ys[: i + 1]
                    X = X[: i + 1]
                    k = i + 1
                    break

            if k <= 0:
                break

            # Sort by objective (minimization)
            order = np.argsort(ys)
            X_sorted = X[order]
            y_sorted = ys[order]

            # Weighted recombination for new mean
            X_sel = X_sorted[:mu]
            # If we used fewer than mu due to last eval, recompute weights for existing size
            if X_sel.shape[0] < mu:
                mu2 = X_sel.shape[0]
                ranks2 = np.arange(mu2, dtype=float)
                raw_w2 = np.log(mu2 + 0.5) - np.log(ranks2 + 1.0)
                w2 = raw_w2 / np.sum(raw_w2)
                x_new = np.sum(X_sel * w2[:, None], axis=0)
                weights_used = w2
            else:
                x_new = np.sum(X_sel * w[:, None], axis=0)
                weights_used = w

            # Sigma adaptation (monotonic improvement heuristic with slight noise-robustness)
            current_best_gen = float(y_sorted[0])
            if current_best_gen + 1e-15 < prev_best_y:
                no_improve_streak = 0
                prev_best_y = current_best_gen
                # Encourage exploration
                sigma *= np.exp(c_sigma / d_sigma * (0.2))
                coord_steps = 0
            else:
                no_improve_streak += 1
                sigma *= np.exp(-c_sigma / d_sigma * (0.35))
                coord_steps += 1

            # Detect boundary pressure: how many candidates hit bounds frequently
            # (Clip ratio approximates infeasible exploration.)
            hit_lo = np.isclose(X_sorted, lo[None, :], rtol=0, atol=1e-12)
            hit_hi = np.isclose(X_sorted, hi[None, :], rtol=0, atol=1e-12)
            boundary_hits = np.mean(hit_lo | hit_hi)
            if boundary_hits > 0.25:
                sigma *= 0.8

            # Diagonal covariance update from selected samples' deviations
            # Use weighted squared deviations from x_new (diagonal only).
            diffs = X_sel - x_new[None, :]
            w_use = weights_used
            # If X_sel has fewer than mu, ensure w_use matches shape
            if w_use.shape[0] != diffs.shape[0]:
                # Recompute quick weights for size
                mu2 = diffs.shape[0]
                ranks2 = np.arange(mu2, dtype=float)
                raw_w2 = np.log(mu2 + 0.5) - np.log(ranks2 + 1.0)
                w_use = raw_w2 / np.sum(raw_w2)

            var_est = np.sum((diffs**2) * w_use[:, None], axis=0) / max(sigma**2, eps_min)
            # Update diag_cov smoothly, with damping and lower bound to keep exploration alive
            diag_cov = (1.0 - c_cov) * diag_cov + c_cov * np.maximum(var_est, 0.05)

            # Commit mean update
            x_mean = np.clip(x_new, lo, hi)

            # Exploitation/exploration switch: coordinate fallback when stalled
            if no_improve_streak >= 6 and coord_steps >= 2 and remaining > 0:
                # Small local search along the best-found coordinate directions.
                # Choose direction where we have largest domain and move from mean.
                # Try a couple of step radii; keep it budget-safe.
                coord_steps_try = 1
                if remaining >= 2:
                    coord_steps_try = 2

                # Select a dimension to perturb: use largest diag_cov scaling as likely useful
                j = int(np.argmax(diag_cov))
                # Radii relative to span and sigma
                radii = [
                    min(0.5 * span[j], 0.5 * sigma),
                    min(0.25 * span[j], 0.25 * sigma),
                ][:coord_steps_try]
                for r in radii:
                    if remaining <= 0:
                        break
                    for sign in (-1.0, 1.0):
                        if remaining <= 0:
                            break
                        x_cand = x_mean.copy()
                        x_cand[j] = np.clip(x_cand[j] + sign * r, lo[j], hi[j])
                        y_c = func(x_cand)
                        remaining -= 1
                        y_c = float(y_c)
                        if y_c < best_y:
                            best_y = y_c
                            best_x = np.array(x_cand, dtype=float, copy=True)
                            x_mean = np.array(x_cand, dtype=float, copy=True)
                            no_improve_streak = 0
                            # Slightly increase local covariance to exploit
                            diag_cov[j] *= 1.2
                        else:
                            diag_cov[j] *= 0.9
                # Reduce sigma a bit after coordinate attempts
                sigma = max(sigma * 0.85, eps_min)

            # Hard guard against numerical issues
            sigma = max(float(sigma), eps_min)
            diag_cov = np.maximum(diag_cov, 1e-8)

            # If sigma is tiny, allow exit only via budget; mean still feasible
            if remaining <= 0:
                break

        return best_x, best_y
