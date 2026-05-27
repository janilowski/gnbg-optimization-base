# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact, budget-aware black-box optimizer for
# minimization on a bounded continuous domain. It is a simple evolutionary
# strategy (ES) with self-adaptive step sizes, combined with local Gaussian
# refinement around the current best solution.
#
# Search state: The algorithm maintains a population of candidate solutions
# (vectors of length dim), their objective values, and per-candidate step
# sizes (sigmas). It also keeps track of the incumbent best point found so far.
#
# Candidate generation: Each iteration creates offspring by sampling Gaussian
# perturbations around existing individuals: x_off = x + sigma * N(0, I).
# Sigmas are self-adapted by multiplicative log-normal noise.
#
# Selection and replacement: The algorithm applies (μ+λ) selection:
# among parents and offspring, the best μ individuals (lowest objective values)
# survive. The global incumbent is updated whenever a better point is found.
#
# Adaptation: Step sizes (sigmas) adapt per individual using log-normal
# multipliers, and a gentle global sigma adjustment is applied based on how
# often offspring improve over parents.
#
# Exploration mechanisms: Random population sampling with adaptive sigmas,
# occasional larger mutations via a mild "restart" impulse when progress stalls.
#
# Exploitation mechanisms: If an improvement is found, the algorithm performs
# local refinement by sampling additional points near the current best with
# a smaller effective step size.
#
# Boundary handling: All candidates are clipped to the provided bounds after
# mutation/refinement, ensuring feasible points always. This handles both
# infinite and finite bounds safely (non-finite bounds are treated as unbounded).
#
# Budget strategy: The total number of function evaluations is capped by the
# provided budget. The algorithm carefully allocates evaluations across
# initialization, ES iterations, and optional local refinement so it never
# exceeds the budget.
#
# Closest known influences: A simplified (μ+λ)-ES with self-adaptation of
# mutation step sizes, plus a small local search around the incumbent best.
#
# Novelty or unusual aspects: The code includes a lightweight, deterministic
# budget accounting mechanism and combines global ES with an on-demand local
# Gaussian refinement triggered by recent improvements.
#
# Failure modes: If the objective is extremely noisy or highly ill-conditioned,
# the selection pressure and step-size adaptation may converge prematurely or
# fail to find improvements within the budget. Clipping at boundaries can also
# distort gradients in boundary-heavy problems.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # --- Read bounds from func ---
        lb = ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)

        if lb is None or ub is None:
            # Fallback: unbounded box in absence of explicit bounds.
            lb = np.full(dim, -np.inf, dtype=float)
            ub = np.full(dim, np.inf, dtype=float)

        # Ensure correct shapes
        lb = lb.reshape(dim)
        ub = ub.reshape(dim)

        # Replace inf bounds with large finite values for stable sampling/clipping.
        # If both are infinite, clipping becomes a no-op.
        finite_lb = np.isfinite(lb)
        finite_ub = np.isfinite(ub)

        # Choose fallback scale based on typical distances; if none available, use 1.
        span = np.where(finite_lb & finite_ub, ub - lb, 1.0)
        finite_span = span[np.isfinite(span)]
        scale = float(np.median(finite_span)) if finite_span.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        clip_lb = np.where(finite_lb, lb, -10.0 * scale)
        clip_ub = np.where(finite_ub, ub, 10.0 * scale)

        def clip_x(x):
            # x: (dim,) float array
            return np.minimum(np.maximum(x, clip_lb), clip_ub)

        # --- Evaluation wrapper with strict budget accounting ---
        evals = 0
        best_x = None
        best_y = np.inf

        def eval_one(x):
            nonlocal evals, best_x, best_y
            if evals >= budget:
                return best_y
            y = float(func(np.asarray(x, dtype=float)))
            evals += 1
            if y < best_y:
                best_y = y
                best_x = np.asarray(x, dtype=float).copy()
            return y

        # --- Initialization ---
        # Population sizes chosen to fit the budget across dimensions.
        # Keep μ small for budget efficiency; λ modest for diversity.
        # Ensure at least 2 evaluations if budget allows.
        if budget <= 0:
            # No evaluations: return a zero vector and +inf.
            return np.zeros(dim, dtype=float), np.inf

        mu = int(max(2, min(8, budget // 2 if budget >= 2 else 2)))
        lam = int(max(2, min(16, max(4, budget - mu) // max(1, mu))))
        # At least one ES generation if budget permits.
        iters = max(1, (budget - mu) // (mu + lam) + 1)

        # Create initial population near the center of the domain (or random in box if needed)
        center = np.where(finite_lb & finite_ub, (lb + ub) / 2.0, np.zeros(dim, dtype=float))

        # Initial sigma: fraction of box span where finite, else global scale.
        init_span = np.where(finite_lb & finite_ub, ub - lb, 2.0 * scale)
        init_sigma = 0.2 * float(np.median(init_span[np.isfinite(init_span)]) if np.any(np.isfinite(init_span)) else 2.0 * scale)
        if not np.isfinite(init_sigma) or init_sigma <= 0:
            init_sigma = 0.5 * scale

        X = np.empty((mu, dim), dtype=float)
        sigmas = np.full(mu, init_sigma, dtype=float)
        ys = np.empty(mu, dtype=float)

        for i in range(mu):
            # Sample uniformly in bounds when finite, else around center with wide Gaussian.
            u = np.random.rand(dim)
            x = np.where(finite_lb & finite_ub, clip_lb + u * (clip_ub - clip_lb), center + init_sigma * np.random.randn(dim))
            x = clip_x(x)
            X[i] = x
            ys[i] = eval_one(x)
            if evals >= budget:
                return best_x if best_x is not None else X[0], best_y

        # Strategy parameters for self-adaptation (log-normal).
        # Standard-ish defaults: tau' and tau.
        tau_prime = 1.0 / np.sqrt(2.0 * dim) if dim > 0 else 0.0
        tau = 1.0 / np.sqrt(2.0 * np.sqrt(dim)) if dim > 0 else 0.0

        # Recombination: simple elitist (pick parents by tournament).
        def tournament_pick(k=3):
            idx = np.random.randint(0, mu, size=k)
            # Lower y is better
            best = idx[np.argmin(ys[idx])]
            return int(best)

        # Track progress for stall detection / local refinement trigger
        best_y_prev = best_y
        stall = 0

        # Main ES loop (budget aware)
        while evals < budget:
            # If next full generation won't fit, do a smaller batch and then stop.
            # We'll cap offspring evaluations.
            remaining = budget - evals
            # If remaining is too small, we may still do one local refinement.
            if remaining <= 0:
                break

            # Determine number of offspring we can evaluate this generation
            # Each offspring needs 1 eval.
            cur_lam = min(lam, remaining)
            if cur_lam <= 0:
                break

            # Create offspring
            X_off = np.empty((cur_lam, dim), dtype=float)
            sig_off = np.empty(cur_lam, dtype=float)
            y_off = np.empty(cur_lam, dtype=float)

            improved_count = 0

            for j in range(cur_lam):
                p = tournament_pick(k=3)
                parent = X[p]

                # Self-adapt sigma with log-normal mutation
                # sigma' = sigma * exp(N(0, tau') + N(0, tau))
                global_noise = np.random.randn() * tau_prime
                local_noise = np.random.randn() * tau
                sigma_new = sigmas[p] * float(np.exp(global_noise + local_noise))
                # Clamp sigma to reasonable range based on box size
                # Avoid collapse to zero or explosion.
                min_sig = 1e-12 * scale
                max_sig = 5.0 * max(1.0, scale)
                if not np.isfinite(sigma_new):
                    sigma_new = init_sigma
                sigma_new = float(np.clip(sigma_new, min_sig, max_sig))

                # Gaussian mutation
                x = parent + sigma_new * np.random.randn(dim)
                x = clip_x(x)

                X_off[j] = x
                sig_off[j] = sigma_new

                y = eval_one(x)
                y_off[j] = y
                if y < best_y_prev:
                    improved_count += 1

                if evals >= budget:
                    # Fill remaining arrays minimally and break
                    # (we'll truncate later)
                    cur_lam = j + 1
                    X_off = X_off[:cur_lam]
                    sig_off = sig_off[:cur_lam]
                    y_off = y_off[:cur_lam]
                    break

            if cur_lam <= 0:
                break

            # Combine parents and offspring
            # (μ+λ) selection: choose best μ individuals from union.
            X_all = np.vstack([X, X_off])
            sig_all = np.hstack([sigmas, sig_off])
            y_all = np.hstack([ys, y_off])

            # Select indices of best mu
            order = np.argsort(y_all)
            keep = order[:mu]
            X = X_all[keep]
            sigmas = sig_all[keep]
            ys = y_all[keep]

            # Update global progress / stall and potential local refinement
            if best_y < best_y_prev - 1e-12:
                stall = 0
            else:
                stall += 1
            best_y_prev = best_y

            # Gentle global sigma adaptation based on success rate
            # Success rate: improvements among offspring relative to previous best.
            success_rate = improved_count / max(1, cur_lam)
            if success_rate > 0.2:
                sigmas *= 1.2
            elif success_rate < 0.05:
                sigmas *= 0.85
            sigmas = np.clip(sigmas, 1e-12 * scale, 5.0 * max(1.0, scale))

            # Exploration impulse on stall: broaden sigmas slightly for diversity
            if stall >= 3 and evals < budget:
                # Restart worst fraction around center
                restart_n = max(1, mu // 3)
                worst = np.argsort(ys)[-restart_n:]
                for idx in worst:
                    if evals >= budget:
                        break
                    # Put near center with larger sigma
                    x = center + (1.5 + np.random.rand(dim)) * scale * 0.3 * np.random.randn(dim)
                    x = clip_x(x)
                    X[idx] = x
                    sigmas[idx] = init_sigma * 1.5
                    ys[idx] = eval_one(x)
                stall = 0  # avoid repeated impulses

            # On-demand local refinement near the incumbent best
            # Trigger if just improved or if we are stalling.
            do_refine = (improved_count > 0) or (stall >= 2)
            if do_refine and best_x is not None and evals < budget:
                remaining = budget - evals
                # Small number of refinement evaluations
                refine_k = min(4, remaining)
                # Local scale decreases over time; use best sigma proxy.
                best_sigma_proxy = float(np.median(sigmas))
                local_sigma = max(1e-12 * scale, 0.35 * best_sigma_proxy)
                # Sample a few points around best_x
                for _ in range(refine_k):
                    if evals >= budget:
                        break
                    step = local_sigma * np.random.randn(dim)
                    x = clip_x(best_x + step)
                    eval_one(x)
                # Further slight reduction after local search to encourage exploitation
                sigmas *= 0.98

            # If we used all budget, exit
            if evals >= budget:
                break

            # Prevent infinite loop in degenerate cases (shouldn't happen)
            iters -= 1
            if iters <= 0:
                break

        # Ensure best_x is valid
        if best_x is None:
            # Evaluate a single point if somehow we never evaluated (e.g., mu=0).
            x0 = clip_x(center + init_sigma * np.random.randn(dim))
            best_x = x0
            best_y = eval_one(x0)

        return best_x, best_y
