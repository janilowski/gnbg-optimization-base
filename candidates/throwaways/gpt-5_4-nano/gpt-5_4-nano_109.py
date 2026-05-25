import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, evaluation-budgeted black-box minimizer inspired by
# a CMA-ES-like evolution strategy with diagonal covariance adaptation and a
# restart mechanism. It works for any dimension and uses only function queries.
# Search state: Maintains a mean vector (mu), an adaptive per-coordinate step size
# (sigma), and a diagonal covariance proxy (var) updated from sampled candidates.
# Candidate generation: Each generation samples offspring from a diagonal Gaussian
# around the mean: x = mu + (sigma * sqrt(var)) * N(0, I). Candidates are clipped
# to the provided bounds.
# Selection and replacement: Samples are evaluated, sorted by objective value, and
# the mean is updated using the best few (recombination) to bias the search downhill.
# Adaptation: The diagonal covariance proxy (var) and sigma are adapted using the
# improvement signal from selected elites; step size increases when progress stalls
# and decreases when improvements are consistent.
# Exploration mechanisms: During the search, a portion of samples are drawn with
# larger effective variance (via a temporary multiplier) to encourage exploration.
# Exploitation mechanisms: The main update uses elite recombination, which focuses
# sampling around promising regions; var is shrunk towards the spread of elites.
# Boundary handling: After sampling, candidates are clipped to [lb, ub]. Mean is
# also kept within bounds to avoid drifting outside.
# Budget strategy: The algorithm strictly tracks remaining evaluations and never
# calls the objective more than the provided budget. The number of generations is
# determined from the budget and offspring batch size, with the last generation
# truncated to fit exactly.
# Closest known influences: Diagonal CMA-ES / evolution strategies with elite recombination,
# step-size adaptation, and restart-on-stagnation behavior.
# Novelty or unusual aspects: Uses a simple diagonal variance proxy with a robust
# clipping-aware adaptation heuristic and supports both (lower/upper) and bounds.lb/ub
# attribute conventions.
# Failure modes: If the objective is extremely noisy or bounds are very tight,
# clipping can cause reduced effective movement; the restart helps mitigate stagnation.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        if self.budget <= 0:
            # No evaluations allowed; best_x/best_y are not meaningful.
            # Return a feasible point at the center of bounds if available, else zeros.
            lb, ub = _get_bounds(func, dim)
            x0 = (lb + ub) / 2.0
            return x0, float("inf")

        lb, ub = _get_bounds(func, dim)
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)

        # Handle degenerate bounds
        span = ub - lb
        span = np.where(span > 0, span, 1.0)

        # Initialize mean at center; initial sigma based on bounds scale.
        mu = np.clip((lb + ub) / 2.0, lb, ub)

        # Choose a batch size that stays robust across dimensions.
        # Typical ES rules: lambda ~ 4 + 3*log(d); cap to budget.
        lam = int(max(8, 4 + 3 * np.log(max(2, dim))))
        lam = min(lam, self.budget)

        # sigma: fraction of span; for high dims, make slightly smaller.
        sigma = 0.3 * float(np.mean(span)) / max(1.0, np.sqrt(dim))
        sigma = max(sigma, 1e-12)

        # Diagonal covariance proxy (all ones initially); var acts like relative spread.
        var = np.ones(dim, dtype=float)

        # Elite recombination count.
        k = max(2, min(lam, int(max(2, lam // 3))))

        # Stagnation tracking for adaptive exploration and restarts.
        best_y = float("inf")
        best_x = mu.copy()

        evals_used = 0
        # Determine number of full generations plus partial last one.
        remaining = self.budget

        # Restart state: if stagnation, reinitialize mean with a random point.
        no_improve_gen = 0
        best_improve = float("inf")
        # Limits
        max_restarts = 3
        restart_count = 0

        # Generation budget
        # Use full lambdas when possible, but allow last partial batch.
        while remaining > 0:
            # If remaining < lam, reduce batch size for last generation.
            cur_lam = min(lam, remaining)
            remaining -= cur_lam

            # Effective variance multiplier for exploration.
            # Increase exploration when stagnating.
            if no_improve_gen >= 5:
                explore_mult = 2.0
            else:
                explore_mult = 1.25 if no_improve_gen >= 2 else 1.0

            # Sample: x = mu + sigma * sqrt(var) * z
            # Mix some exploratory samples by temporarily scaling var.
            sqrt_var = np.sqrt(np.maximum(var, 1e-18))
            z = np.random.randn(cur_lam, dim)

            # Decide which individuals are exploratory.
            if explore_mult != 1.0 and cur_lam >= 4:
                # Roughly 25% exploratory
                exp_mask = np.random.rand(cur_lam) < 0.25
                z[exp_mask] *= np.sqrt(explore_mult)
            # Evaluate candidates
            xs = mu + (sigma * sqrt_var) * z
            xs = np.clip(xs, lb, ub)

            ys = np.empty(cur_lam, dtype=float)
            for i in range(cur_lam):
                y = func(xs[i])
                ys[i] = float(y)
            evals_used += cur_lam

            # Update global best
            idx_best = int(np.argmin(ys))
            if ys[idx_best] < best_y:
                best_y = float(ys[idx_best])
                best_x = xs[idx_best].copy()
                no_improve_gen = 0
                best_improve = best_y
            else:
                no_improve_gen += 1

            # Sort by fitness (minimization)
            order = np.argsort(ys)
            elite = xs[order[:k]]
            elite_y = ys[order[:k]]

            # Recombination weights (favor best more).
            # Use linear decreasing weights normalized.
            weights = np.linspace(1.0, 0.2, k, dtype=float)
            weights /= np.sum(weights)
            new_mu = np.sum(elite * weights[:, None], axis=0)
            new_mu = np.clip(new_mu, lb, ub)

            # Update variance proxy based on elite spread around elite-weighted mean.
            # The factor is normalized to keep var bounded.
            centered = elite - new_mu
            # Weighted second moment
            cov_diag = np.sum((centered ** 2) * weights[:, None], axis=0)
            # Avoid zeros; var scales to typical elite dispersion relative to sigma.
            cov_diag = np.maximum(cov_diag, 1e-18)

            # Convert cov_diag to relative var: cov_diag ≈ (sigma^2 * var)
            rel_var = cov_diag / (sigma ** 2 + 1e-30)
            rel_var = np.clip(rel_var, 1e-6, 1e6)

            # Adapt var with smoothing
            lr_var = 0.35 if dim <= 20 else 0.25
            var = (1.0 - lr_var) * var + lr_var * rel_var

            # Step-size adaptation:
            # If best improves in this generation (relative), decrease sigma; otherwise increase slightly.
            gen_best = float(elite_y[0])
            improvement = best_y - gen_best  # after update best_y already maybe smaller; use alternative:
            # We can estimate progress by comparing elite mean to previous best_y.
            elite_mean = float(np.mean(elite_y))
            # Use a stable progress proxy:
            progress = (best_y - elite_mean)
            # If elite_mean is much smaller than current best_y, we likely improved previously.
            if progress > 0:
                # shrink
                sigma *= 0.82
            else:
                # if no good offspring, broaden
                sigma *= 1.05

            sigma = float(np.clip(sigma, 1e-12, 1e6))

            # Mean update (with mild pull towards best_x to intensify)
            # Intensify if elite is very good.
            mu = 0.75 * mu + 0.25 * new_mu
            # Additional pull towards global best once close to optimum
            if np.isfinite(best_y):
                # If current elite best is near global best, exploit
                if gen_best <= best_y * (1.0 + 1e-6):
                    mu = 0.9 * mu + 0.1 * best_x
            mu = np.clip(mu, lb, ub)

            # Restart logic on stagnation: if too many gens without improvement, reinitialize.
            if no_improve_gen >= 10:
                restart_count += 1
                if restart_count > max_restarts:
                    # Reduce sigma aggressively to attempt a final exploitation.
                    sigma *= 0.6
                    no_improve_gen = 0
                else:
                    # Random reset around a feasible point.
                    # Choose random point within bounds, but center near current best if available.
                    t = np.random.rand(dim)
                    x_rand = lb + t * (ub - lb)
                    if np.isfinite(best_y):
                        # Move mean toward best_x with some randomness
                        alpha = 0.5
                        mu = np.clip(alpha * best_x + (1 - alpha) * x_rand, lb, ub)
                    else:
                        mu = x_rand

                    var = np.ones(dim, dtype=float)
                    # Reset sigma to a fraction of span
                    sigma = 0.25 * float(np.mean(span)) / max(1.0, np.sqrt(dim))
                    sigma = max(sigma, 1e-12)
                    no_improve_gen = 0

        # The loop respects budget by reducing `remaining` by cur_lam each generation.
        # evals_used should be <= budget.
        # Return best found.
        if not np.isfinite(best_y):
            # If objective returned NaN everywhere, fall back to mean.
            best_y = float("inf")
            best_x = mu.copy()
        return best_x, best_y


def _get_bounds(func, dim):
    """
    Retrieve bounds from either:
      - func.lower / func.upper
      - func.bounds.lb / func.bounds.ub
    Returns (lb, ub) as numpy arrays of shape (dim,).
    """
    # Try lower/upper
    if hasattr(func, "lower") and hasattr(func, "upper"):
        lb = np.asarray(getattr(func, "lower"), dtype=float)
        ub = np.asarray(getattr(func, "upper"), dtype=float)
        lb, ub = _broadcast_bounds(lb, ub, dim)
        return lb, ub

    # Try nested bounds
    if hasattr(func, "bounds"):
        b = getattr(func, "bounds")
        if hasattr(b, "lb") and hasattr(b, "ub"):
            lb = np.asarray(getattr(b, "lb"), dtype=float)
            ub = np.asarray(getattr(b, "ub"), dtype=float)
            lb, ub = _broadcast_bounds(lb, ub, dim)
            return lb, ub

    # If bounds missing, assume wide default box
    # (Still feasible for many benchmarks; harness likely supplies bounds.)
    lb = -5.0 * np.ones(dim, dtype=float)
    ub = 5.0 * np.ones(dim, dtype=float)
    return lb, ub


def _broadcast_bounds(lb, ub, dim):
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    if lb.size == 1:
        lb = np.full(dim, float(lb[0]), dtype=float)
    if ub.size == 1:
        ub = np.full(dim, float(ub[0]), dtype=float)
    if lb.size != dim or ub.size != dim:
        # Best-effort: if one side mismatches, attempt trunc/pad with last value.
        lb = _fit_length(lb, dim)
        ub = _fit_length(ub, dim)
    # Ensure lb <= ub
    swap = lb > ub
    if np.any(swap):
        lo = lb.copy()
        lb[swap] = ub[swap]
        ub[swap] = lo[swap]
    return lb, ub


def _fit_length(arr, dim):
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size == dim:
        return arr
    if arr.size == 0:
        return np.zeros(dim, dtype=float)
    if arr.size > dim:
        return arr[:dim]
    # pad with last value
    pad_val = arr[-1]
    out = np.empty(dim, dtype=float)
    out[: arr.size] = arr
    out[arr.size :] = pad_val
    return out
