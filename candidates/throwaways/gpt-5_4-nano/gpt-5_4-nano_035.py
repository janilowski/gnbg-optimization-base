# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# using a simple global-to-local evolutionary strategy (ES) with dynamic
# step-size control and restarts. It works for continuous domains and uses
# only function evaluations via the provided callable interface.
#
# Search state: The algorithm maintains a current population of candidate
# points, their objective values, a best-so-far solution, and a mutation
# step size (sigma). It also tracks how many evaluations have been consumed.
#
# Candidate generation: Each generation creates offspring by sampling
# isotropic Gaussian perturbations around the current population members.
# Offspring are clipped to the problem bounds to respect constraints.
#
# Selection and replacement: After evaluating offspring, it performs a
# (μ, λ) selection: the next population is formed from the best individuals
# among the current + offspring set, keeping population size constant.
#
# Adaptation: Sigma is adapted using a successful/unsuccessful rule based on
# whether the best value improved within a generation.
#
# Exploration mechanisms: Multiple restarts are performed when progress
# stalls, re-centering the population near the best known point and
# increasing sigma to escape local basins.
#
# Exploitation mechanisms: When improvements occur, sigma is decreased and
# the search focuses around the best solutions, refining the local optimum.
#
# Boundary handling: Candidates are clipped to [lb, ub] (with robust handling
# for degenerate intervals where lb == ub).
#
# Budget strategy: The algorithm never exceeds the given evaluation budget.
# It uses a preplanned number of generations and truncates work for the final
# generation to fit the remaining evaluations.
#
# Closest known influences: The behavior is reminiscent of a simple evolution
# strategy / CMA-ES-like intuition (population mutation with step-size control)
# but deliberately remains lightweight and isotropic for compactness.
#
# Novelty or unusual aspects: Uses a blend of μ, λ selection with an
# improvement-based sigma adaptation and budget-aware truncation. It also
# supports multiple bound attribute formats as requested.
#
# Failure modes: For extremely high-dimensional or very noisy objectives,
# isotropic sampling may be inefficient. If the feasible region is narrow or
# bounds are degenerate, clipping can reduce effective diversity.
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
            # No evaluations allowed; return something deterministic within bounds
            lb, ub = _get_bounds(func, dim)
            x = lb.copy()
            return x, float("inf")

        lb, ub = _get_bounds(func, dim)

        # Ensure numeric arrays with proper shapes
        lb = np.asarray(lb, dtype=float).reshape(dim)
        ub = np.asarray(ub, dtype=float).reshape(dim)

        # If bounds are degenerate, sampling must collapse to a single point.
        span = ub - lb
        degenerate = span <= 0.0
        if np.all(degenerate):
            x0 = lb.copy()
            y0 = float(func(x0))
            return x0, y0

        # ---- Parameters (kept dimension-robust) ----
        # Population size grows gently with dimension but is bounded to keep budget use efficient.
        mu = max(4, min(20, 4 + dim // 2))
        lam = max(mu, min(60, mu * 2))
        # Total per generation: lam evaluations
        evals_used = 0

        def clip_to_bounds(x):
            if np.any(degenerate):
                # For degenerate dimensions, force exact lb/ub (they are equal or invalid).
                x = np.where(degenerate, lb, x)
            return np.minimum(ub, np.maximum(lb, x))

        # Initial center: uniform random point in bounds (or mid if no span)
        x_center = clip_to_bounds(lb + span * np.random.rand(dim))

        def eval_one(x):
            nonlocal evals_used
            if evals_used >= budget:
                # Should never happen due to budget-aware scheduling; guard anyway.
                return float("inf")
            y = func(np.asarray(x, dtype=float))
            evals_used += 1
            return float(y)

        # Initial population around x_center
        # Initialize sigma as a fraction of average span magnitude.
        avg_span = float(np.mean(np.maximum(span, 0.0)))
        # Fallback if spans are extremely small
        sigma = 0.2 * avg_span if avg_span > 0 else 0.1

        # Create initial population (mu individuals), evaluate them.
        pop = np.empty((mu, dim), dtype=float)
        for i in range(mu):
            if avg_span > 0:
                # Mix between center and uniform to avoid early collapse.
                t = 0.3 + 0.7 * np.random.rand()
                xi = (1.0 - t) * x_center + t * (lb + span * np.random.rand(dim))
                # Add small gaussian noise
                xi = xi + np.random.randn(dim) * (0.05 * avg_span)
            else:
                xi = x_center.copy()
            pop[i] = clip_to_bounds(xi)

        pop_y = np.empty(mu, dtype=float)
        for i in range(mu):
            if evals_used >= budget:
                # Not enough budget for full initial population; evaluate until exhausted.
                pop_y[i] = float("inf")
            else:
                pop_y[i] = eval_one(pop[i])

        # Track best-so-far
        best_idx = int(np.argmin(pop_y))
        best_x = pop[best_idx].copy()
        best_y = float(pop_y[best_idx])

        # Generation scheduling
        # We will spend lam evals per full generation, but cap by budget.
        # Determine number of full generations possible after initial evaluations.
        remaining = budget - evals_used
        # If budget is very small, just return current best.
        if remaining <= 0:
            return best_x, best_y

        # Rough number of generations: at least 1 if budget permits.
        # We'll handle final partial generation by truncating offspring count.
        gens = max(1, remaining // lam)
        # Allow a few extra generations if budget allows (but still safe due to truncation).
        gens = min(gens + 2, 2000)

        # Restart / stagnation logic
        stall = 0
        stall_limit = max(5, 1 + dim // 5)
        # Improvement thresholds
        improve_tol = 1e-12
        best_y_prev = best_y

        # Selection uses μ best from offspring combined with population (simple ES style).
        # Also compute an "average parent" to center the next mutation distribution.
        for g in range(gens):
            if evals_used >= budget:
                break

            # Determine offspring count for this generation respecting remaining evaluations.
            remaining = budget - evals_used
            if remaining <= 0:
                break
            k = min(lam, remaining)  # number of evaluations this generation

            # Choose parents: best μ in current population
            order = np.argsort(pop_y)
            parents = pop[order[:mu]].copy()

            # Occasionally re-center on the best point for exploitation
            # and on mean of parents for robustness.
            if k >= lam:
                # Use both contributions
                mean_par = np.mean(parents, axis=0)
                x_center = 0.5 * mean_par + 0.5 * best_x
            else:
                # In partial generations, center more strongly on the best
                x_center = 0.75 * best_x + 0.25 * np.mean(parents, axis=0)

            # Adapt sigma softly each generation based on bounds span
            # Keep sigma within reasonable range to avoid stagnation.
            # Upper cap based on average span; lower cap on numeric stability.
            sigma_max = (0.5 * avg_span) if avg_span > 0 else 1.0
            sigma_min = 1e-12 if avg_span > 0 else 1e-6
            sigma = float(np.clip(sigma, sigma_min, max(sigma_max, sigma_min)))

            # Generate offspring around parents (isotropic Gaussian)
            offspring = np.empty((k, dim), dtype=float)
            for i in range(k):
                # Pick a parent (biased towards better indices)
                # Use exponential bias for robustness.
                # Probability proportional to exp(-rank)
                rank = np.random.randint(0, mu)
                if mu > 1:
                    # Bias sampling to better parents
                    if np.random.rand() < 0.7:
                        # pick better half more often
                        rank = np.random.randint(0, max(1, mu // 2))
                p = parents[rank]

                # Mutation: x = p + N(0, sigma^2 I)
                # Additional component from x_center to improve mixing.
                step = np.random.randn(dim) * sigma
                mix = 0.2 + 0.8 * np.random.rand()
                xi = (1.0 - mix) * p + mix * x_center + step

                offspring[i] = clip_to_bounds(xi)

            # Evaluate offspring
            off_y = np.empty(k, dtype=float)
            for i in range(k):
                off_y[i] = eval_one(offspring[i])

            # Combine and select best μ (replacement)
            combined = np.vstack((pop, offspring))
            combined_y = np.concatenate((pop_y, off_y))
            # Only consider finite values if any (due to budget guard)
            valid = np.isfinite(combined_y)
            if not np.any(valid):
                break

            order = np.argsort(np.where(valid, combined_y, np.inf))
            new_pop = combined[order[:mu]].copy()
            new_pop_y = combined_y[order[:mu]].astype(float)

            pop = new_pop
            pop_y = new_pop_y

            # Update best-so-far
            idx = int(np.argmin(pop_y))
            y_best_gen = float(pop_y[idx])
            if y_best_gen < best_y - improve_tol:
                best_y = y_best_gen
                best_x = pop[idx].copy()
                # Successful: decrease sigma to exploit
                sigma *= 0.82
                stall = 0
            else:
                # Unsuccessful: increase sigma to explore
                sigma *= 1.08
                stall += 1

            # Stagnation restart: reinitialize population around best_x
            if stall >= stall_limit and evals_used < budget:
                stall = 0
                # Increase sigma to escape local optimum
                sigma = min(sigma_max if sigma_max > 0 else 1.0, max(sigma, 0.5 * (avg_span if avg_span > 0 else 1.0)))
                # Re-sample population around best_x (small gaussian + uniform blend)
                for i in range(mu):
                    if evals_used >= budget:
                        break
                    if avg_span > 0:
                        blend = 0.4 + 0.6 * np.random.rand()
                        u = lb + span * np.random.rand(dim)
                        xi = (1.0 - blend) * best_x + blend * u
                        xi = xi + np.random.randn(dim) * (0.1 * sigma)
                    else:
                        xi = best_x.copy()
                    pop[i] = clip_to_bounds(xi)
                    pop_y[i] = eval_one(pop[i])
                idx = int(np.argmin(pop_y))
                if pop_y[idx] < best_y - improve_tol:
                    best_y = float(pop_y[idx])
                    best_x = pop[idx].copy()

            best_y_prev = best_y_prev  # no-op to keep structure readable

        return best_x, best_y


def _get_bounds(func, dim):
    """
    Read bounds from func.lower/func.upper or func.bounds.lb/func.bounds.ub.
    Returns (lb, ub) as numpy arrays of shape (dim,).
    """
    # Option 1: func.lower / func.upper
    if hasattr(func, "lower") and hasattr(func, "upper"):
        lb = np.asarray(getattr(func, "lower"), dtype=float).reshape(-1)
        ub = np.asarray(getattr(func, "upper"), dtype=float).reshape(-1)
        if lb.size != dim or ub.size != dim:
            # Allow scalar bounds
            if lb.size == 1 and ub.size == 1:
                lb = np.full(dim, float(lb.item()), dtype=float)
                ub = np.full(dim, float(ub.item()), dtype=float)
            else:
                raise ValueError("Bounds lower/upper must match dim or be scalar.")
        return lb, ub

    # Option 2: func.bounds.lb / func.bounds.ub
    if hasattr(func, "bounds"):
        b = getattr(func, "bounds")
        if hasattr(b, "lb") and hasattr(b, "ub"):
            lb = np.asarray(getattr(b, "lb"), dtype=float).reshape(-1)
            ub = np.asarray(getattr(b, "ub"), dtype=float).reshape(-1)
            if lb.size != dim or ub.size != dim:
                if lb.size == 1 and ub.size == 1:
                    lb = np.full(dim, float(lb.item()), dtype=float)
                    ub = np.full(dim, float(ub.item()), dtype=float)
                else:
                    raise ValueError("Bounds bounds.lb/bounds.ub must match dim or be scalar.")
            return lb, ub

    raise AttributeError("Cannot read bounds: expected func.lower/func.upper or func.bounds.lb/func.bounds.ub.")
