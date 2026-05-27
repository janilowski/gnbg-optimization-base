# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy inspired by
# evolutionary search with a local "coordinate direction" refinement step.
# It maintains a small population of candidate solutions and iteratively
# improves them by sampling around the current best with adaptive step sizes.
# Search state: Keeps a population of vectors X, their objective values Y,
# current evaluation count, and an adaptive global step scale sigma. Also
# tracks the incumbent best solution found so far.
# Candidate generation: Each iteration creates multiple offspring by adding
# Gaussian noise scaled by sigma to existing population members, plus one
# directed refinement proposal that probes along coordinate-aligned directions
# derived from recent improvements.
# Selection and replacement: Uses elitist replacement—combines parents and
# offspring, then keeps the best individuals (lowest objective values).
# Adaptation: sigma is increased slightly when improvements stall and decreased
# when strong improvements occur, using success signals from the current
# generation.
# Exploration mechanisms: Population-based Gaussian sampling, including
# occasional larger-radius steps to escape local minima.
# Exploitation mechanisms: A local refinement step around the current best
# using coordinate-aligned perturbations; if it improves, sigma shrinks.
# Boundary handling: Proposed points are clipped to the provided bounds
# (supports func.lower/upper or func.bounds.lb/ub).
# Budget strategy: The algorithm strictly counts objective evaluations and
# stops when the given budget is reached (never exceeds it). The number of
# offspring per iteration is capped to fit remaining budget.
# Closest known influences: Lightweight variants of (μ+λ)-ES and coordinate
# probing are combined into a robust, dimension-agnostic loop.
# Novelty or unusual aspects: A simple coordinate-aligned refinement direction
# is inferred from the best-so-far movement and used as a targeted proposal.
# Failure modes: If the objective is extremely noisy or highly ill-conditioned,
# the adaptive sigma may oscillate; clipping can also cause stagnation on
# narrow feasible regions. If bounds are very tight, the refinement step
# may have little effect.
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
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/ub")

        if lb.shape == ():  # scalar bound
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        lb = np.broadcast_to(lb, (dim,)).astype(float, copy=False)
        ub = np.broadcast_to(ub, (dim,)).astype(float, copy=False)

        # Ensure well-formed bounds
        if np.any(ub < lb):
            raise ValueError("Invalid bounds: some ub < lb")

        span = ub - lb
        # If span is zero in any dimension, that coordinate is fixed.
        span_safe = np.where(span > 0, span, 1.0)

        # Strict evaluation counter
        evals = 0
        n = dim

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_one(x):
            nonlocal evals
            if evals >= budget:
                # Should never happen due to budget checks.
                return float("inf")
            evals += 1
            y = func(np.asarray(x, dtype=float))
            return float(y)

        # --- Initialization ---
        # Population size based on dimension and budget; keep small but effective.
        # Must be able to evaluate at least one candidate.
        # Default: between 4 and 12 (or fewer if budget is tight).
        lam = int(min(max(4, n + 1), 12))
        if budget < lam:
            lam = max(1, budget)
        mu = int(max(2, lam // 2))

        # Sample initial points uniformly within bounds.
        # If bounds are degenerate, all samples collapse to the same point.
        X = lb + np.random.random((lam, n)) * span_safe
        X = clip(X)

        Y = np.empty(lam, dtype=float)
        for i in range(lam):
            if evals >= budget:
                break
            Y[i] = eval_one(X[i])

        # Sort by fitness (minimization)
        idx = np.argsort(Y)
        X = X[idx]
        Y = Y[idx]

        best_x = X[0].copy()
        best_y = float(Y[0])

        # Adaptive step size: start as a fraction of the box width.
        # Use global sigma to keep compact.
        sigma = 0.25 * np.mean(span_safe)
        sigma = max(sigma, 1e-12)

        # Track last best movement to build a coordinate direction.
        last_best = best_x.copy()

        # --- Main loop: iterate generating offspring until budget exhaustion ---
        # Each generation creates k offspring and uses elitist (μ+λ) replacement.
        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break

            # Determine offspring count within remaining budget.
            # Keep k >= 1 when possible.
            k = int(min(max(1, lam), remaining))
            # Use parents from top mu
            parents = X[:mu]

            # Success tracking for sigma adaptation
            improved = 0

            # --- Candidate generation (exploration + exploitation) ---
            # Offspring:
            #  - Gaussian around parents
            #  - One targeted coordinate refinement around best
            offspring = []
            offspring_y = []

            # Exploration: sample k-1 points (if possible) around random parents
            # with isotropic Gaussian noise scaled by sigma.
            num_explore = max(0, k - 1)

            if num_explore > 0:
                # Choose parent indices uniformly among top mu
                p_idx = np.random.randint(0, mu, size=num_explore)
                P = parents[p_idx]  # shape (num_explore, n)

                # Gaussian noise; also add slight anisotropy by scaling with span
                # so dimensions with larger feasible ranges explore more.
                noise = np.random.randn(num_explore, n)
                # Scale noise by span_safe to normalize across dimensions.
                scale = (span_safe / (np.mean(span_safe) + 1e-15)).reshape(1, -1)
                Z = P + (sigma * noise * scale)
                Z = clip(Z)

                # Evaluate
                for i in range(num_explore):
                    if evals >= budget:
                        break
                    y = eval_one(Z[i])
                    offspring.append(Z[i])
                    offspring_y.append(y)

            # Exploitation: directed coordinate probing near best
            # Build a direction vector from movement; fall back to ones if none.
            if evals < budget and len(offspring) < k:
                d = best_x - last_best
                if not np.any(np.isfinite(d)):
                    d = np.zeros(n)
                # Determine coordinate with largest absolute movement
                if np.allclose(d, 0.0):
                    # Use a fixed axis direction: random coordinate
                    j = int(np.random.randint(0, n))
                    dir_vec = np.zeros(n)
                    dir_vec[j] = 1.0
                else:
                    j = int(np.argmax(np.abs(d)))
                    dir_vec = np.zeros(n)
                    dir_vec[j] = np.sign(d[j]) if d[j] != 0 else 1.0

                # Probe two-sided; pick one that uses available budget
                step = sigma
                cand1 = clip(best_x + step * dir_vec)
                cand2 = clip(best_x - step * dir_vec)

                # Evaluate up to one (keeps budget simpler)
                chosen = cand1
                y = eval_one(chosen)
                offspring.append(chosen)
                offspring_y.append(y)

            if not offspring:
                break

            offspring = np.asarray(offspring, dtype=float)
            offspring_y = np.asarray(offspring_y, dtype=float)

            # Count improvements over incumbent
            improved = int(np.sum(offspring_y < best_y))
            if improved > 0:
                jbest = int(np.argmin(offspring_y))
                if float(offspring_y[jbest]) < best_y:
                    best_y = float(offspring_y[jbest])
                    best_x = offspring[jbest].copy()

            # --- Selection and replacement (elitist) ---
            # Combine parents and offspring, keep best lam individuals.
            X_comb = np.vstack([X, offspring])
            Y_comb = np.concatenate([Y, offspring_y])

            order = np.argsort(Y_comb)
            X = X_comb[order][:lam]
            Y = Y_comb[order][:lam]

            last_best = last_best.copy()
            # Update last_best towards best_x if it improved (else keep)
            if best_x is not None:
                last_best = best_x.copy() if np.random.random() < 0.25 else last_best

            # --- Adaptation of sigma ---
            # If we improved, shrink sigma; else slowly grow to encourage exploration.
            # Also incorporate normalized improvement magnitude.
            best_off = float(np.min(offspring_y))
            delta = best_off - best_y  # negative if offspring produced new best (approximately)
            # Use relative signal with scale of box.
            rel_scale = (np.mean(span_safe) + 1e-15)
            signal = (best_off - best_y) / rel_scale  # <= 0 if improved

            if improved > 0 and signal <= 0:
                sigma *= 0.85
            else:
                sigma *= 1.05

            # Occasional larger jump to escape stagnation
            if np.random.random() < 0.05 and evals < budget:
                sigma *= 1.25

            # Keep sigma within reasonable range
            sigma = float(np.clip(sigma, 1e-12, 0.5 * np.max(span_safe) + 1e-12))

            if evals >= budget:
                break

        return best_x, best_y
