# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# for bounded continuous domains. It uses a population-based evolution strategy
# with coordinate-wise step sizes and occasional global restart behavior.
# Search state: The algorithm maintains a small population of candidate points,
# their objective values, and per-dimension mutation scales (step sizes).
# Candidate generation: Each generation creates offspring by sampling Gaussian
# perturbations around selected parents. The step size vector is also
# adapted based on how successful moves are.
# Selection and replacement: The best individuals are selected (elitist
# selection). The population is replaced by a mix of elites and new offspring
# to keep progress while maintaining diversity.
# Adaptation: The coordinate-wise step sizes are increased when offspring
# improve and decreased when they do not, using a smoothed success rate.
# Exploration mechanisms: Gaussian sampling provides exploration; diversity is
# reinforced with occasional larger-variance "exploratory" offspring.
# Exploitation mechanisms: Elites and success-driven smaller step sizes focus
# sampling near promising regions.
# Boundary handling: Candidates are clipped to the provided bounds; step sizes
# are limited to avoid wasting evaluations far outside the feasible region.
# Budget strategy: The total number of objective evaluations is capped using
# a strict counter; each evaluation calls func(x) exactly once. When the budget
# is nearly exhausted, the algorithm stops and returns the best point seen.
# Closest known influences: The behavior is reminiscent of CMA-ES/ES style
# adaptation, but implemented compactly with diagonal (coordinate-wise) updates.
# Novelty or unusual aspects: The diagonal step-size adaptation uses a
# success-rate controller per dimension, combined with occasional "restart-like"
# resampling when progress stalls.
# Failure modes: If the objective is extremely noisy or deceptive, the success
# controller may over-shrink steps; restarts mitigate this. If bounds are very
# tight, clipping may cause many identical points and reduce effective search.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds ----
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("func must provide bounds via lower/upper or bounds.lb/bounds.ub")

        if lb.shape == () and dim != 1:
            lb = np.full(dim, float(lb))
        if ub.shape == () and dim != 1:
            ub = np.full(dim, float(ub))

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds must match dim")

        # Ensure proper ordering
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        span = hi - lo
        span = np.where(span <= 0, 1.0, span)  # avoid divide-by-zero; degenerate dims will be clipped anyway

        def clip01(x):
            return np.minimum(hi, np.maximum(lo, x))

        # ---- Evaluation wrapper with strict budget ----
        evals = 0

        def f(x):
            nonlocal evals
            if evals >= budget:
                # Must not exceed budget; return +inf so it won't be selected.
                return float("inf")
            y = func(np.asarray(x, dtype=float))
            evals += 1
            return float(y)

        # ---- Initialize population ----
        # Population size: keep small for budget efficiency; at least 4.
        # Ensure not exceeding budget: at least one parent must exist.
        base_pop = 4 + int(3 * np.log(max(dim, 2)))
        pop_size = min(max(base_pop, 4), max(4, budget // 2) if budget >= 8 else max(4, budget))
        pop_size = min(pop_size, budget)  # avoid impossible cases

        # Start points uniform in bounds
        # If span is tiny, all points will be very similar, but that's expected.
        X = lo + np.random.random((pop_size, dim)) * (hi - lo)

        Y = np.empty(pop_size, dtype=float)
        best_idx = 0
        best_y = float("inf")
        for i in range(pop_size):
            Y[i] = f(X[i])
            if Y[i] < best_y:
                best_y = Y[i]
                best_idx = i
            if evals >= budget:
                return X[best_idx].copy(), best_y

        # Step sizes: diagonal, based on bounds and typical scale.
        # Start at ~10% of span or smaller depending on dim.
        # Keep them within a sensible range.
        init_sigma = 0.1 * span / (1.0 + np.sqrt(dim))
        sigma = np.maximum(init_sigma, 1e-12)

        # Success tracking (smoothed)
        success_rate = np.full(dim, 0.0, dtype=float)
        target_success = 0.2  # typical ES target
        c_succ = 0.2  # smoothing factor

        # Elite selection size
        elite_k = max(1, pop_size // 4)

        # Progress/stagnation tracking for occasional reseed
        last_best = best_y
        stagnation = 0
        best_x = X[best_idx].copy()

        # ---- Main loop ----
        # Each generation uses (pop_size - elite_k) new evaluations to replace worst.
        # Keep evaluations under budget by truncating remaining offspring count.
        while evals < budget:
            # Determine how many offspring we can still afford
            remaining = budget - evals
            # New points needed this iteration: pop_size - elite_k
            need_offspring = pop_size - elite_k
            if need_offspring <= 0:
                break
            n_offspring = min(need_offspring, remaining)
            if n_offspring <= 0:
                break

            # Sort by fitness (minimization)
            order = np.argsort(Y)
            X = X[order]
            Y = Y[order]

            # Keep elites
            elites = X[:elite_k].copy()
            elite_y = Y[:elite_k].copy()

            # Best parent guiding exploitation
            parent_best = elites[0]

            # Exploration/exploitation mixture:
            # With some probability and/or when stagnating, sample around best with larger noise.
            # Otherwise sample around a randomly chosen elite.
            stagnate_bonus = 1.0 + 2.0 * (stagnation >= 5)
            # Exploratory fraction increases when stagnating
            explore_frac = 0.3 + 0.2 * (stagnation >= 3)

            # Create offspring
            X_new = np.empty((n_offspring, dim), dtype=float)
            sigma_used = np.empty((n_offspring, dim), dtype=float)

            for j in range(n_offspring):
                if np.random.random() < explore_frac:
                    base = parent_best
                    # Larger variance for exploration
                    scale = (1.0 + 1.5 * np.random.random()) * stagnate_bonus
                else:
                    # Choose a random elite as base
                    base = elites[np.random.randint(elite_k)]
                    scale = 1.0 + 0.8 * np.random.random()

                # Diagonal Gaussian perturbation
                step = sigma * scale
                eps = np.random.normal(loc=0.0, scale=1.0, size=dim)
                x = base + eps * step
                x = clip01(x)

                X_new[j] = x
                sigma_used[j] = step

            # Evaluate offspring
            Y_new = np.empty(n_offspring, dtype=float)
            for j in range(n_offspring):
                Y_new[j] = f(X_new[j])

            # Merge populations: elites + offspring, then keep best pop_size
            X_merged = np.vstack([elites, X_new])
            Y_merged = np.concatenate([elite_y, Y_new])
            ord2 = np.argsort(Y_merged)
            X = X_merged[ord2][:pop_size]
            Y = Y_merged[ord2][:pop_size]

            # Update best trackers
            if Y[0] < best_y - 1e-15:
                best_y = Y[0]
                best_x = X[0].copy()
                improvement = True
            else:
                improvement = False

            if improvement:
                stagnation = 0
            else:
                stagnation += 1
                # If long stagnation, do a mild diagonal reset to escape
                if stagnation >= 8:
                    # Reinflate sigma based on span to re-explore
                    sigma = np.minimum(span * (0.25 / (1.0 + np.sqrt(dim))), sigma * 2.0)
                    success_rate *= 0.5
                    stagnation = 0

            # ---- Adaptation using per-dimension success rate ----
            # Compare offspring to current best elite value (or overall best).
            # A coordinate is considered "successful" if the child improved and that
            # coordinate moved in the direction consistent with the perturbation.
            # Since we only have diagonal noise, we approximate with improvement boolean
            # and movement magnitude per coordinate.
            #
            # For robustness, update success_rate based on improvement and normalized
            # step usage for each coordinate among the new evaluated points.
            if n_offspring > 0:
                # Determine which offspring were improvements over current best elite_y[0]
                threshold = elites[0]  # not used; keep clarity
                current_best = elites[0]  # x vector
                current_best_y = elite_y[0]

                improved_mask = Y_new < current_best_y
                if np.any(improved_mask):
                    # Compute movement magnitude for improved points
                    # Estimate per-coordinate normalized movement relative to sigma_used.
                    # Then map to a success signal.
                    moved = X_new[improved_mask] - current_best
                    sig = sigma_used[improved_mask]
                    # Avoid divide-by-zero: cap minimum sig
                    sig = np.maximum(sig, 1e-12)
                    norm_move = np.abs(moved) / sig
                    # Convert to a bounded success indicator: higher norm_move slightly more likely success
                    # but keep within [0, 1] via saturation.
                    succ_signal = 1.0 - np.exp(-norm_move)
                    # Average across improved offspring
                    succ_avg = np.mean(succ_signal, axis=0)
                    # Smooth update
                    success_rate = (1.0 - c_succ) * success_rate + c_succ * succ_avg
                else:
                    # No improvements: decay success_rate toward 0
                    success_rate = (1.0 - c_succ) * success_rate

            # Step size update:
            # If success_rate > target -> increase; else decrease.
            # Use multiplicative log-normal style update for stability.
            # Per coordinate adaptation (diagonal ES-ish behavior).
            inc = (success_rate > target_success).astype(float)
            dec = 1.0 - inc

            # Magnitudes: small gain
            gain = 0.15
            # Increase factor when success exceeds target; otherwise decrease.
            sigma = sigma * np.exp(gain * inc - gain * dec)
            # Keep sigma within reasonable limits relative to span
            sigma = np.maximum(sigma, 1e-12)
            sigma = np.minimum(sigma, 0.5 * span)

        # Final return
        # Ensure we return best known point within budget.
        # If budget is extremely small, best_x already tracked.
        return best_x.copy(), float(best_y)
