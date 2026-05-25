# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer for continuous
# search spaces using a variant of the Covariance Matrix Adaptation evolution
# strategy (CMA-ES) with diagonal covariance. The algorithm minimizes the given
# objective and returns the best point found within the evaluation budget.
#
# Search state: Maintains a population of candidate solutions each iteration,
# tracks the current best solution, maintains a mean vector (search center),
# and maintains per-dimension step sizes (diagonal covariance).
#
# Candidate generation: Each iteration samples lambda candidate points from a
# multivariate normal distribution centered at the current mean with standard
# deviations given by the diagonal step sizes. Sampling uses numpy's RNG.
#
# Selection and replacement: Evaluates all candidates, selects the mu best
# individuals (lowest objective values), and updates the mean as their weighted
# average.
#
# Adaptation: Updates diagonal step sizes using a simplified CMA-like rule based
# on the improvement/selection signal and an evolution path proxy. Covariance
# is kept diagonal for robustness and compactness.
#
# Exploration mechanisms: Population sampling around the mean with nonzero
# step sizes; step sizes adapt over time to maintain exploration.
#
# Exploitation mechanisms: Weighted recombination of the best individuals shifts
# the mean toward promising regions and gradually reduces step sizes as progress
# is detected.
#
# Boundary handling: Samples are clipped to provided box bounds (lower/upper)
# before evaluation. Mean updates are also clamped to bounds to stay feasible.
#
# Budget strategy: Uses the provided evaluation budget strictly. It performs
# as many full iterations as possible with remaining evaluations, and if a
# partial iteration is needed for the last batch it evaluates only the required
# number of candidates.
#
# Closest known influences: Inspired by CMA-ES: weighted recombination and adaptive
# step-size/covariance updates; simplified to diagonal covariance for compactness.
#
# Novelty or unusual aspects: Uses a conservative diagonal adaptation with
# clipped sampling to maintain feasibility under hard box constraints while
# preserving CMA-like selection pressure.
#
# Failure modes: In very high dimensions or with highly ill-conditioned
# landscapes, diagonal covariance may adapt slowly; clipping at bounds can lead
# to stagnation. If the objective is noisy, selection-based updates may cause
# premature contraction, but step sizes tend to recover due to sampling noise.
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
            # No evaluations allowed; return a deterministic feasible point.
            lb, ub = _get_bounds(func, dim)
            x0 = np.where(np.isfinite(lb) & np.isfinite(ub), (lb + ub) / 2.0, np.zeros(dim))
            y0 = float("inf")
            return x0, y0

        lb, ub = _get_bounds(func, dim)
        lb = lb.astype(float, copy=False)
        ub = ub.astype(float, copy=False)
        if dim == 0:
            return np.array([], dtype=float), float("inf")

        # Population size and recombination weights (CMA-ES style)
        # lambda should be >= 4 for robustness; scale with dimension.
        lam = int(max(4, 4 + 3 * np.log(dim + 1.0)))
        mu = lam // 2
        if mu < 1:
            mu = 1

        # Logarithmic weights: w_i ~ log(mu+0.5) - log(i)
        ranks = np.arange(mu, dtype=float)
        weights = np.log(mu + 0.5) - np.log(ranks + 1.0)
        weights = weights / np.sum(weights)

        # Initialize mean at center of bounds if finite, else at 0.
        finite = np.isfinite(lb) & np.isfinite(ub)
        mean = np.zeros(dim, dtype=float)
        if np.any(finite):
            mean[finite] = 0.5 * (lb[finite] + ub[finite])
        # Initialize diagonal step sizes: fraction of range or 1.
        sigma = np.ones(dim, dtype=float)
        if np.any(finite):
            range_ = (ub - lb)
            # Avoid zero range; if range too small, keep moderate sigma.
            base = np.where(range_ > 0, range_, 1.0)
            sigma = 0.3 * base / np.sqrt(dim)
        sigma = np.maximum(sigma, 1e-12)

        # Best tracking
        best_x = mean.copy()
        best_y = float("inf")

        evals = 0

        # Learning rates (simplified diagonal CMA)
        # These hyperparameters control adaptation speed.
        # Typically c_sigma relates to step-size control; keep conservative.
        c_sigma = 0.3
        d_sigma = 1.0 + dim ** 0.5
        # For diagonal adaptation, use rank-mu update strength.
        c_cov = 0.2

        # Evolution path proxy for step adaptation (diagonal).
        # This helps produce smoother contraction/expansion.
        p_sigma = np.zeros(dim, dtype=float)

        # Helper to sample one population batch
        def sample_population(k):
            # x = mean + sigma * N(0,1) (diagonal covariance)
            # For robustness: work in standardized coordinates then scale.
            z = np.random.randn(k, dim)
            y = mean + z * sigma  # elementwise broadcasting
            y = np.clip(y, lb, ub)
            return y, z

        # Iteration loop with strict budget accounting
        while evals < budget:
            remaining = budget - evals
            # Evaluate up to lam candidates (or fewer if remaining budget)
            k = lam if remaining >= lam else remaining
            xs, zs = sample_population(k)

            # Evaluate candidates
            ys = np.empty(k, dtype=float)
            for i in range(k):
                val = func(xs[i])
                # Ensure float conversion (objective might return numpy scalar)
                ys[i] = float(val)

            evals += k

            # Update best
            idx_best = int(np.argmin(ys))
            if ys[idx_best] < best_y:
                best_y = ys[idx_best]
                best_x = xs[idx_best].copy()

            # Sort by fitness (minimization)
            order = np.argsort(ys)
            x_sel = xs[order[:mu]]
            z_sel = zs[order[:mu]]

            # Weighted recombination: mean <- sum w_i x_i
            old_mean = mean.copy()
            mean = (weights[:, None] * x_sel).sum(axis=0)

            # Clamp mean to bounds for feasibility (especially when bounds are tight)
            mean = np.clip(mean, lb, ub)

            # Update step-size adaptation (diagonal):
            # Use the normalized improvement direction via selected steps.
            # Compute mean shift in standardized coordinates:
            # p_sigma <- (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)) * inv_sqrt_sigma * (mean-old_mean)/sigma
            # Here inv_sqrt_sigma is 1 (since sigma is per-dim).
            diff = mean - old_mean
            # Standardized direction: diff / sigma
            standardized = diff / np.maximum(sigma, 1e-30)
            # Smooth evolution path proxy
            p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma)) * standardized

            # Rank-mu covariance proxy for diagonal:
            # Use weighted average of squared standardized selected z's.
            # This estimates how "useful" dimensions are.
            # (Since diagonal, just use per-dim variances.)
            # weighted_z2 = sum w_i * z_i^2
            z2 = z_sel * z_sel
            weighted_z2 = (weights[:, None] * z2).sum(axis=0)

            # Step-size update: sigma <- sigma * exp( (c_cov*(weighted_z2-1) + (p_sigma^2 - 1)/d_sigma ) / something )
            # The exponent form helps keep sigma positive.
            # Note: weighted_z2 around 1 indicates neutral; >1 expands, <1 contracts.
            neutral = 1.0
            # p_sigma^2 signal (per-dim). Center around 1 as rough proxy for progress.
            p_signal = (p_sigma * p_sigma)
            # Compose update; keep conservative magnitude.
            update = c_cov * (weighted_z2 - neutral) + (p_signal - neutral) / max(d_sigma, 1.0)
            # Mild scaling to avoid aggressive changes
            sigma = sigma * np.exp(np.clip(update, -1.0, 1.0) * 0.2)

            # Ensure sigma doesn't collapse completely; also limit growth to bounds scale.
            # Approximate max sigma as range / 2 if finite; else keep reasonable.
            if np.any(np.isfinite(lb) & np.isfinite(ub)):
                max_sigma = 0.5 * np.maximum(ub - lb, 1e-12) / np.sqrt(dim)
                max_sigma = np.where(np.isfinite(max_sigma), max_sigma, sigma)
                sigma = np.minimum(sigma, np.maximum(max_sigma, 1e-12))
            sigma = np.maximum(sigma, 1e-12)

        return best_x, best_y


def _get_bounds(func, dim):
    # Prefer func.lower/func.upper, else func.bounds.lb/ub.
    lb = None
    ub = None

    if hasattr(func, "lower") and hasattr(func, "upper"):
        lb = np.asarray(func.lower)
        ub = np.asarray(func.upper)
    elif hasattr(func, "bounds"):
        b = func.bounds
        if hasattr(b, "lb") and hasattr(b, "ub"):
            lb = np.asarray(b.lb)
            ub = np.asarray(b.ub)

    if lb is None or ub is None:
        # If bounds unavailable, use +/- inf (clipping becomes no-op for inf).
        lb = np.full(dim, -np.inf, dtype=float)
        ub = np.full(dim, np.inf, dtype=float)
        return lb, ub

    # Broadcast/reshape to dim
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)

    if lb.size != dim or ub.size != dim:
        # If bounds are scalar, broadcast; otherwise fall back to inf for safety.
        if lb.size == 1 and ub.size == 1:
            lb = np.full(dim, float(lb[0]), dtype=float)
            ub = np.full(dim, float(ub[0]), dtype=float)
        else:
            # Best-effort: clip to min common size, then pad with inf.
            minsz = min(lb.size, ub.size, dim)
            lb2 = np.full(dim, -np.inf, dtype=float)
            ub2 = np.full(dim, np.inf, dtype=float)
            lb2[:minsz] = lb[:minsz]
            ub2[:minsz] = ub[:minsz]
            lb, ub = lb2, ub2

    # Ensure lb <= ub where possible; swap if reversed.
    swap = np.isfinite(lb) & np.isfinite(ub) & (lb > ub)
    if np.any(swap):
        tmp = lb[swap].copy()
        lb[swap] = ub[swap]
        ub[swap] = tmp

    return lb, ub
