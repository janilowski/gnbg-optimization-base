# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer using a
# population-based evolution strategy hybridized with coordinate-wise local
# search. The algorithm works for any dimension >= 1 and respects the provided
# evaluation budget exactly.
# Search state: Maintains a small population of candidate points plus their
# objective values, tracking the global best found so far.
# Candidate generation: Starts with a random population uniformly sampled
# within the provided bounds. Each iteration proposes new candidates by adding
# Gaussian perturbations around the current best and around top individuals,
# combined with occasional coordinate-wise mutations to adapt across dimensions.
# Selection and replacement: Uses elitist selection (mu+lambda style): merges
# parents and offspring, ranks by objective value (minimization), and keeps
# the best mu points as the next population; the global best is updated.
# Adaptation: The mutation step-size sigma is adapted based on the improvement
# observed over recent iterations, using a simple success-based rule.
# Exploration mechanisms: Population diversity via random re-sampling and
# Gaussian/global perturbations, plus coordinate-wise mutations to escape
# axis-aligned traps.
# Exploitation mechanisms: Stronger sampling near the current best using a
# decreasing step-size schedule and additional local coordinate refinement.
# Boundary handling: Any candidate leaving the bounds is clamped to the nearest
# bound; if clamping causes duplicates, diversity is restored by slight jitter.
# Budget strategy: Computes the number of evaluations allowed per iteration and
# never calls the objective more than budget times. The final evaluation budget
# is allocated to iterative updates until exhausted.
# Closest known influences: Inspired by basic evolution strategies (ES) and
# success-rule step-size adaptation, combined with a simple local search around
# the incumbent best.
# Novelty or unusual aspects: Uses a dimension-aware coordinate mutation
# probability and a small deterministic local refinement schedule when progress
# stalls.
# Failure modes: On extremely noisy objectives or very tight bounds, step-size
# adaptation may converge prematurely; clamping may reduce effective diversity.
# The algorithm still remains budget-safe and returns the best-so-far.
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
            # No evaluations allowed; return a feasible point as placeholder.
            lb, ub = self._get_bounds(func)
            x0 = np.clip(np.zeros(dim, dtype=float), lb, ub)
            return x0, float("inf")

        lb, ub = self._get_bounds(func)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        span = ub - lb
        span = np.where(np.isfinite(span), span, 1.0)
        span = np.where(span == 0.0, 1.0, span)

        rng = np.random

        # Parameters: kept small for compactness and robustness.
        mu = max(2, min(8, 2 + dim // 2))      # parents kept
        pop = mu                                # offspring count each iteration (lambda=mu)
        min_it = 1

        # Allocate evaluations:
        # We'll evaluate an initial population of mu candidates, then iterate with
        # pop offspring each iteration while leaving room for final updates.
        # Total exact evaluation budget tracking is enforced below.
        evals_used = 0

        # Helper to clamp and avoid NaNs/infs.
        def clamp(x):
            x = np.asarray(x, dtype=float)
            # Replace nan/inf with midpoints to be safe.
            bad = ~np.isfinite(x)
            if np.any(bad):
                mid = lb + 0.5 * (ub - lb)
                x[bad] = mid[bad]
            return np.clip(x, lb, ub)

        def sample_uniform(n):
            # Sample uniformly in bounds; handles infinite bounds by fallback.
            # If lb/ub are infinite, uniform sampling isn't well-defined; but the
            # harness is expected to provide finite bounds.
            if np.any(~np.isfinite(lb)) or np.any(~np.isfinite(ub)):
                # Fall back to a standard normal around 0 scaled by span.
                mid = np.where(np.isfinite(lb) & np.isfinite(ub), 0.5 * (lb + ub), 0.0)
                x = mid + rng.randn(n, dim) * span
                return clamp(x)
            r = rng.rand(n, dim)
            return lb + r * (ub - lb)

        def evaluate_batch(X):
            nonlocal evals_used
            ys = np.empty(X.shape[0], dtype=float)
            for i in range(X.shape[0]):
                if evals_used >= budget:
                    # Stop immediately; caller must avoid requesting more.
                    ys = ys[:i]
                    return ys
                x = clamp(X[i])
                ys[i] = func(x)
                evals_used += 1
            return ys

        def unique_jitter(X, scale):
            # If many duplicates due to clamping, add tiny jitter to restore diversity.
            # scale is relative to span.
            X = X.copy()
            # Jitter magnitude:
            eps = 1e-12 + 1e-6
            jitter = rng.randn(*X.shape) * (eps * scale)
            return clamp(X + jitter)

        # Step-size initialization: proportional to domain size.
        sigma0 = 0.25 * np.mean(span) if np.any(span > 0) else 1.0
        sigma = float(sigma0) if sigma0 > 0 else 1.0

        # Initial population
        n_init = min(mu, budget)
        X = sample_uniform(n_init)
        # If bounds are very tight, diversity might be poor; add slight jitter.
        if n_init > 1:
            X = unique_jitter(X, span)
        y = evaluate_batch(X)

        # Make sure we always keep arrays with consistent shapes.
        if y.shape[0] != X.shape[0]:
            # Budget exhausted during init; return best so far.
            k = int(np.argmin(y))
            return clamp(X[k]), float(y[k])

        # Maintain parents: best mu points (may be smaller near budget limit).
        order = np.argsort(y)
        X = X[order]
        y = y[order]
        if X.shape[0] > mu:
            X = X[:mu]
            y = y[:mu]

        best_idx = int(np.argmin(y))
        best_x = clamp(X[best_idx])
        best_y = float(y[best_idx])

        # Iteration count derived from remaining budget
        remaining = budget - evals_used
        if remaining <= 0:
            return best_x, best_y

        # How many full iterations we can do, given pop offspring each time.
        # Ensure at least one iteration when possible.
        max_iters = max(min_it, remaining // pop) if pop > 0 else 0
        # We'll also allow a final partial iteration to use leftover budget exactly.
        # Track a simple improvement counter for sigma adaptation.
        no_improve = 0
        prev_best = best_y

        # Coordinate mutation probability increases when dim grows.
        p_coord = min(0.5, 0.3 + 0.05 * (dim / 10.0))

        for _ in range(max_iters + 2):  # +2 to allow final partial iteration
            if evals_used >= budget:
                break

            # Determine offspring count for this round (exactly within budget).
            remain = budget - evals_used
            lam = min(pop, remain)
            if lam <= 0:
                break

            # Selection base: pick from top individuals and bias toward best_x.
            # Build offspring with mixture of:
            #  - global perturbation around best
            #  - around other elites
            #  - occasional coordinate-wise mutation
            # Choose parent indices.
            elite_count = min(len(X), mu)
            if elite_count <= 0:
                break

            # Mixture weights
            w_best = 0.65
            w_elite = 0.35

            # Create offspring
            offspring = np.empty((lam, dim), dtype=float)

            # A decreasing schedule encourages exploitation later.
            t = evals_used / max(1, budget)
            exploit_scale = (1.0 - t) ** 0.7
            local_sigma = max(1e-12, sigma * (0.6 + 0.9 * exploit_scale))

            for i in range(lam):
                if rng.rand() < w_best:
                    center = best_x
                else:
                    idx = rng.randint(0, elite_count)
                    center = X[idx]

                # Gaussian step, possibly anisotropic based on center distance to best.
                step = rng.randn(dim)
                # With some probability, do coordinate-wise mutation instead of full Gaussian.
                if rng.rand() < (p_coord * (0.25 + 0.75 * (1.0 - exploit_scale))):
                    x_new = center.copy()
                    # Mutate a subset of coordinates
                    m = max(1, int(rng.randint(1, dim + 1) * (0.15 + 0.35 * rng.rand())))
                    coords = rng.choice(dim, size=m, replace=False)
                    # Step per chosen coordinate
                    x_new[coords] = x_new[coords] + (local_sigma * 0.6) * step[coords]
                    # Remaining coordinates get a tiny perturbation to avoid stalling
                    other = np.setdiff1d(np.arange(dim), coords, assume_unique=False)
                    if other.size > 0 and rng.rand() < 0.2:
                        x_new[other] = x_new[other] + (local_sigma * 0.05) * rng.randn(other.size)
                    offspring[i] = x_new
                else:
                    # Full Gaussian move with a small heavy-tail component.
                    # heavy_tail via occasional large factor
                    if rng.rand() < 0.08:
                        factor = 1.0 + 5.0 * abs(rng.randn())
                    else:
                        factor = 1.0
                    offspring[i] = center + (local_sigma * factor) * step

            offspring = clamp(offspring)

            # Avoid exact duplicates (common when bounds are tight and clamping hits).
            # If many duplicates, add tiny jitter.
            if lam > 1:
                # Measure duplication via rounding tolerance
                rounded = np.round(offspring / max(1e-12, np.mean(span)) , 10)
                # Count unique rows
                uniq = np.unique(rounded, axis=0)
                if uniq.shape[0] < max(2, lam // 2):
                    offspring = unique_jitter(offspring, span)

            y_off = evaluate_batch(offspring)

            if y_off.shape[0] == 0:
                break

            # Combine parents and offspring, then select best mu.
            X_comb = np.vstack([X, offspring[: y_off.shape[0]]])
            y_comb = np.concatenate([y, y_off])

            order = np.argsort(y_comb)
            X = X_comb[order][:mu]
            y = y_comb[order][:mu]

            # Update best
            k = int(np.argmin(y))
            if float(y[k]) < best_y - 1e-15:
                best_y = float(y[k])
                best_x = clamp(X[k])
                no_improve = 0
            else:
                no_improve += 1

            # Step-size adaptation: success-based with mild oscillation guard.
            if best_y < prev_best - 1e-15:
                sigma *= 1.05
            else:
                # If no improvement for a couple of rounds, reduce sigma.
                if no_improve >= 2:
                    sigma *= 0.85
                    no_improve = max(0, no_improve - 1)
                else:
                    sigma *= 0.97

            # Optional local coordinate refinement when progress stalls.
            if no_improve >= 3 and evals_used < budget:
                # Evaluate a few neighbor points along random coordinates near best_x.
                # Each refinement costs 2 evaluations max per coordinate step.
                refine_coords = min(dim, max(1, 1 + dim // 4))
                coords = rng.choice(dim, size=refine_coords, replace=False)
                step = max(1e-12, 0.2 * sigma)

                # Construct neighbors (2 per coordinate) but only within remaining budget.
                remain = budget - evals_used
                max_neighbors = min(2 * refine_coords, remain)
                neighbors = np.empty((max_neighbors, dim), dtype=float)
                idx = 0
                for c in coords:
                    if idx >= max_neighbors:
                        break
                    x1 = best_x.copy()
                    x2 = best_x.copy()
                    x1[c] = x1[c] + step
                    x2[c] = x2[c] - step
                    neighbors[idx] = x1
                    idx += 1
                    if idx < max_neighbors:
                        neighbors[idx] = x2
                        idx += 1

                y_nei = evaluate_batch(neighbors[:idx]) if idx > 0 else np.array([])
                if y_nei.size > 0:
                    j = int(np.argmin(y_nei))
                    if float(y_nei[j]) < best_y - 1e-15:
                        best_y = float(y_nei[j])
                        best_x = clamp(neighbors[j])

                        # If we found an improvement, slightly increase sigma to explore.
                        sigma *= 1.03
                        no_improve = 0

            prev_best = best_y

        return best_x, best_y

    @staticmethod
    def _get_bounds(func):
        # Accept bounds via func.lower/upper or func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = getattr(func, "lower")
            ub = getattr(func, "upper")
            return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)

        if hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            # Some harnesses use b.lb / b.ub
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return np.asarray(b.lb, dtype=float), np.asarray(b.ub, dtype=float)
            # Or might use b.lower/upper
            if hasattr(b, "lower") and hasattr(b, "upper"):
                return np.asarray(b.lower, dtype=float), np.asarray(b.upper, dtype=float)

        raise AttributeError(
            "Objective function must provide bounds via func.lower/func.upper "
            "or func.bounds.lb/func.bounds.ub."
        )
