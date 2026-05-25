import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact population-based black-box minimizer (a CMA-ES-inspired
# evolution strategy) that works for continuous box-bounded problems in any dimension.
# Search state: Maintains a mean vector, a global step-size (sigma), and a diagonal
# covariance (implemented as per-coordinate variances via "D"), plus evolution-path
# vectors for adaptation. Tracks the best-so-far solution and its objective value.
# Candidate generation: Each iteration samples a population from a multivariate normal
# distribution with covariance ~ diag(D^2) and global scaling sigma. Samples are
# transformed to the problem space and clipped to stay within bounds.
# Selection and replacement: Evaluates each candidate, selects the best individuals by
# objective (minimization), and updates the mean as a weighted combination of the top
# candidates. The best overall solution is preserved across all iterations.
# Adaptation: Updates step-size using a cumulative path length criterion similar to
# CMA-ES, and updates the diagonal covariance via rank-μ style adaptation using
# normalized steps, keeping the algorithm robust with only diagonal covariance.
# Exploration mechanisms: Initial sigma and covariance adaptation allow wide exploration;
# population sampling provides stochastic global search; step-size adaptation maintains
# appropriate exploration/exploitation balance.
# Exploitation mechanisms: Mean update toward selected low-cost points increases local
# exploitation; reduced step-size and refined diagonal variances tighten the search.
# Boundary handling: Uses objective evaluations for clipped points only (candidates
# are clipped to bounds before evaluation). This preserves feasibility without needing
# special constraints handling.
# Budget strategy: Uses an evaluation budget exactly by tracking remaining evaluations.
# Each iteration evaluates up to a fixed population size or whatever remains; it never
# exceeds the provided budget.
# Closest known influences: CMA-ES (minimizing evolution strategies with step-size control
# and covariance adaptation), simplified to diagonal covariance for compactness and speed.
# Novelty or unusual aspects: Diagonal-only covariance with CMA-like adaptation and
# robust guardrails (eps floors, sanity checks) for numerical stability while remaining
# compact.
# Failure modes: If bounds are extremely tight or objective is extremely noisy, diagonal
# adaptation may stagnate or oscillate; numerical issues are mitigated by floor values.
# For very high dimensions, diagonal approximation may limit performance compared to full
# covariance CMA-ES.
# ALGORITHM_ANALYSIS_NOTE_END

class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim

        # Read bounds from the provided func interface
        lb = getattr(func, "lower", None)
        ub = getattr(func, "upper", None)
        if lb is None or ub is None:
            bounds = getattr(func, "bounds", None)
            if bounds is None:
                raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/ub.")
            lb = getattr(bounds, "lb", None)
            ub = getattr(bounds, "ub", None)

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimension mismatch with dim.")
        # Ensure lb <= ub
        if np.any(ub < lb):
            raise ValueError("Invalid bounds: found upper < lower.")

        # Evaluation budget bookkeeping
        budget = self.budget
        if budget <= 0:
            # Nothing to evaluate; return a feasible point
            x0 = (lb + ub) * 0.5
            return x0, float("inf")

        # Population size and number of iterations based on budget.
        # Use a reasonable default scaling with dimension, but keep evaluations exact.
        lam = int(np.clip(4 + 3 * np.log(dim + 1.0), 8, 32))
        # Ensure we don't exceed budget with even 1 iteration; if budget is tiny, reduce lam.
        lam = max(2, min(lam, budget))
        mu = lam // 2

        # Recombination weights (positive, normalized)
        ranks = np.arange(lam, dtype=float)
        # Use log weights; standard in ES/CMA
        weights = np.log(mu + 0.5) - np.log(ranks[:mu] + 1.0)
        weights = np.maximum(weights, 0.0)
        wsum = np.sum(weights)
        if wsum <= 0:
            weights = np.ones(mu) / mu
        else:
            weights /= wsum
        # Effective selection mass
        mu_eff = 1.0 / np.sum(weights**2)

        # Initialize mean at center, step-size from bounds span
        xmean = (lb + ub) * 0.5
        span = ub - lb
        # Avoid zero span: if bound interval is 0, sigma should be small but nonzero
        sigma = 0.3 * np.max(span)
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 0.1

        # Diagonal covariance: store per-coordinate standard deviations ("D")
        # Start with normalized covariance based on span
        D = np.ones(dim, dtype=float)
        # Normalize initial D by span to make sigma roughly comparable
        # (If span is tiny, avoid division by near-zero.)
        safe_span = np.maximum(span, 1e-12)
        D = safe_span / np.max(safe_span)
        D = np.clip(D, 0.1, 10.0)

        # CMA-like evolution paths (diagonal approximation)
        ps = np.zeros(dim, dtype=float)
        pc = np.zeros(dim, dtype=float)

        # Strategy parameter settings
        # Damping factor for step-size control
        cs = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (dim + 1e-12)) - 1.0) + cs
        cc = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)

        # Learning rate for covariance diagonal update
        c1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        # Rank-μ learning rate portion for covariance update
        cmu = min(1.0 - c1, 2.0 * (mu_eff - 1.0) / ((dim + 2.0) ** 2 + mu_eff))
        # Diagonal-only adaptation uses a simplified form:
        # c1 scales pc update; cmu scales weighted outer products of selected steps.

        chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))
        eps = 1e-12

        best_x = np.clip(xmean, lb, ub)
        best_y = float("inf")

        # Track number of evaluations
        evals_used = 0

        # Helper: clip to bounds
        def clip_to_bounds(x):
            return np.minimum(np.maximum(x, lb), ub)

        # Main loop: each iteration consumes up to lam evaluations
        # Use while to avoid overshooting budget.
        while evals_used < budget:
            remaining = budget - evals_used
            cur_lam = min(lam, remaining)
            # Sample standard normal vectors for diagonal covariance
            # z shape: (cur_lam, dim)
            z = np.random.randn(cur_lam, dim)
            # Transform samples: x = xmean + sigma * (z * D)
            # Here covariance is diag((D*sigma)^2)
            y = xmean + (sigma * (z * D))
            y = clip_to_bounds(y)

            # Evaluate candidates and keep fitness values
            fitness = np.empty(cur_lam, dtype=float)
            for i in range(cur_lam):
                fitness[i] = float(func(y[i]))
            evals_used += cur_lam

            # Update best-so-far
            idx_best = int(np.argmin(fitness))
            if fitness[idx_best] < best_y:
                best_y = float(fitness[idx_best])
                best_x = np.array(y[idx_best], copy=True)

            # Rank selection for recombination
            order = np.argsort(fitness)
            selected_idx = order[:mu]
            x_sel = y[selected_idx]
            z_sel = z[selected_idx]  # corresponds to steps in standard space

            # Update mean as weighted combination of selected points
            xmean_old = xmean
            xmean = np.dot(weights, x_sel)

            # Normalize step in evolution path update:
            # In diagonal CMA, we use inv(D) scaling:
            # y_norm = (xmean - xmean_old) / (sigma * D)
            step = (xmean - xmean_old) / (sigma * (D + eps))
            # Evolution path for sigma
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mu_eff) * step / sqrt(1) (diagonal norm already)
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mu_eff) * step / np.sqrt(dim)

            # Step-size adaptation: sigma *= exp( (cs/ds) * (||ps||/chi_n - 1) )
            ps_norm = np.linalg.norm(ps)
            sigma_factor = np.exp((cs / ds) * (ps_norm / (chi_n + eps) - 1.0))
            # Guard against numerical blow-ups; keep sigma within reasonable range
            sigma = sigma * sigma_factor
            sigma = float(np.clip(sigma, 1e-18, 1e18))

            # Evolution path for covariance: pc = (1-cc)*pc + hsig*sqrt(cc*(2-cc)*mu_eff) * (xmean-xmean_old)/sigma/D
            hsig = 1.0
            # Use a common CMA criterion, adapted to diagonal scaling:
            # if ps_norm / sqrt(1-(1-cs)^(2*t)) < 1.4+2/(dim+1), then hsig=1.
            # We lack t, approximate with ps_norm criterion relative to chi_n.
            if ps_norm / (chi_n + eps) < 2.0:
                hsig = 1.0
            else:
                hsig = 0.0

            pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mu_eff) * step

            # Diagonal covariance update
            # Update D^2 via rank-one and rank-μ contributions.
            # Maintain variances v = D^2.
            v = D * D

            # Rank-one term using pc (diagonal outer): (pc^2 - v) scaled by c1
            c1_term = c1 * (pc * pc)
            v = (1.0 - c1 - cmu) * v + c1_term

            # Rank-μ term: weighted sum of selected z (steps in standard normal space)
            # For diagonal-only, contribution to variance is sum(w_i * z_sel_i^2).
            # Because covariance is diag((sigma*D)^2), in standard space the expected z^2 contributes directly.
            z2 = z_sel * z_sel
            rank_mu = np.dot(weights, z2)  # shape (dim,)
            v = v + cmu * rank_mu

            # Convert back to D with floors for stability
            # Keep D^2 not too small to allow movement, and not too large.
            v = np.clip(v, 1e-24, 1e24)
            D = np.sqrt(v)

        return np.array(best_x, copy=False), float(best_y)
