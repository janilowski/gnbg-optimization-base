# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm (derivative-free)
# using a population of candidate points plus repeated local “restart” phases. It
# maintains a best-so-far solution and uses coordinate-wise perturbations to explore
# while progressively shrinking a step-size based on observed improvements.
# Search state: Stores the current best point (x_best) and best value (y_best),
# a working population around the current best, a per-run step scale (sigma),
# and an evaluation counter to ensure the total budget is never exceeded.
# Candidate generation: Samples candidates using Gaussian perturbations around
# the current best. Additionally, performs coordinate-wise probes (one dimension
# at a time) when the algorithm detects stagnation, using symmetric offsets.
# Selection and replacement: For each iteration, all candidates are evaluated (if
# budget allows). The best candidate replaces the global best. The population is
# then refreshed around the updated best for the next iteration; otherwise, the step
# size shrinks.
# Adaptation: Step-size (sigma) decays when improvements stall. If no improvement is
# observed for several iterations, the algorithm performs a “restart-like” expansion
# (up to a safe maximum) to re-diversify search.
# Exploration mechanisms: Global exploration via sampling around the best with a
# moderate sigma; occasional diversification when stagnation persists.
# Exploitation mechanisms: Local refinement using coordinate-wise symmetric probes and
# repeated sampling with shrinking sigma after improvements.
# Boundary handling: Candidate points are clipped to the provided box constraints.
# Budget strategy: Every objective call increments an internal counter; the algorithm
# ensures it never evaluates more than the provided budget by checking remaining
# evaluations before each batch and before each candidate evaluation.
# Closest known influences: Inspired by evolution-strategy style sampling with step-size
# adaptation and by coordinate-search probes for robustness; implemented in a compact,
# budget-aware manner.
# Novelty or unusual aspects: Combines batch Gaussian sampling with deterministic
# coordinate probes triggered by stagnation, all while strictly enforcing the budget.
# Failure modes: If the objective is extremely noisy or deceptive, the algorithm may
# waste evaluations on unproductive probes; shrinking sigma could slow recovery, but
# restarts/diversification mitigate this. In very high dimensions, coordinate probes
# are costly, so they are used sparingly.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        # Read bounds from either func.lower/func.upper or func.bounds.lb/func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            lb = np.asarray(getattr(b, "lb"), dtype=float)
            ub = np.asarray(getattr(b, "ub"), dtype=float)
        else:
            raise AttributeError("Function must provide bounds via lower/upper or bounds.lb/bounds.ub.")

        lb = np.broadcast_to(lb, (self.dim,))
        ub = np.broadcast_to(ub, (self.dim,))
        if np.any(ub < lb):
            raise ValueError("Invalid bounds: ub must be >= lb for all dimensions.")

        # Ensure finite range handling
        span = ub - lb
        span_safe = np.where(span > 0, span, 1.0)

        # Clip helper
        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        # Budget-aware evaluation wrapper
        evals = 0

        def eval_x(x):
            nonlocal evals
            if evals >= self.budget:
                return None
            evals += 1
            return float(func(np.asarray(x, dtype=float)))

        # Initial point: center with small noise if possible
        x0 = 0.5 * (lb + ub)
        # Use at least one evaluation for a valid best-so-far
        y0 = eval_x(x0)
        # In case budget is 0 (edge case), return something deterministic
        if y0 is None:
            return np.asarray(x0, dtype=float), float("inf")

        x_best = np.asarray(x0, dtype=float).copy()
        y_best = y0

        # Population size (kept modest for standard-library speed)
        # More candidates early helps; keep within budget.
        # Budget might be small; adapt gracefully.
        if self.budget <= 2:
            return x_best, y_best

        # Use a small batch size relative to remaining budget
        base_pop = 8
        pop = min(base_pop, max(2, (self.budget - 1) // 2 + 1))
        pop = int(pop)

        # Initial step size: fraction of domain size
        sigma_max = 0.5 * span_safe
        sigma_min = 1e-12
        sigma = 0.25 * sigma_max

        # Stagnation tracking
        no_improve_count = 0
        max_no_improve = 6

        # Iteration count capped by budget
        # Each loop evaluates up to pop candidates (plus occasional probes)
        # so we derive a conservative upper bound.
        max_iters = max(1, (self.budget - 1) // pop)

        rng = np.random

        for _ in range(max_iters):
            # Remaining budget check
            remaining = self.budget - evals
            if remaining <= 0:
                break

            # Generate population around current best.
            # Mix isotropic and per-coordinate scaling from sigma.
            # Gaussian exploration; clipping ensures feasibility.
            batch_n = min(pop, remaining)
            # To reduce pathological behavior, sample standard normals and scale.
            Z = rng.standard_normal((batch_n, self.dim))
            # Broadcast sigma to each row; sigma is per-dimension.
            X = clip(x_best + Z * sigma)

            # Evaluate batch and select local best
            local_best_x = None
            local_best_y = y_best

            for i in range(batch_n):
                # Strictly never exceed budget: eval_x checks it.
                y = eval_x(X[i])
                if y is None:
                    break
                if y < local_best_y:
                    local_best_y = y
                    local_best_x = X[i].copy()

            # Update global best
            improved = local_best_x is not None and local_best_y < y_best
            if improved:
                x_best = local_best_x
                y_best = local_best_y
                no_improve_count = 0
                # Exploitation: shrink sigma moderately after success (but not too fast)
                sigma = np.maximum(sigma * 0.85, sigma_min)
            else:
                no_improve_count += 1
                # Exploration/exploitation trade-off: shrink if no progress
                sigma = np.maximum(sigma * 0.8, sigma_min)

            # If stagnating, perform coordinate-wise symmetric probes sparsely
            if no_improve_count >= 2 and no_improve_count <= max_no_improve and remaining > 0:
                # Use at most a small probe budget to keep total evaluations bounded.
                # The number of coordinates probed scales with dimension but capped.
                remaining = self.budget - evals
                if remaining <= 0:
                    break

                # Probe count: min of a cap and a dimension-dependent budget.
                # In high-dimensions, probe only few coordinates.
                probe_cap = max(2, min(6, self.dim))
                # Ensure we don't spend too much: each coordinate costs 2 evals.
                probe_cap = min(probe_cap, remaining // 2) if remaining >= 2 else 0
                if probe_cap > 0:
                    # Choose coordinates: random subset biased towards larger spans (more responsive)
                    # Also avoid deterministic selection to reduce aliasing.
                    coord_weights = span_safe / np.maximum(span_safe.mean(), 1e-12)
                    coord_weights = np.maximum(coord_weights, 1e-3)
                    coord_weights = coord_weights / coord_weights.sum()
                    coords = rng.choice(self.dim, size=probe_cap, replace=False, p=coord_weights)

                    # Symmetric offsets along each chosen coordinate
                    # Use a step proportional to current sigma and per-coordinate span.
                    # Ensure non-zero perturbations.
                    for j in coords:
                        if evals >= self.budget:
                            break
                        step = sigma[j]
                        if step <= 0:
                            continue
                        x_plus = x_best.copy()
                        x_minus = x_best.copy()
                        x_plus[j] = np.clip(x_plus[j] + step, lb[j], ub[j])
                        x_minus[j] = np.clip(x_minus[j] - step, lb[j], ub[j])

                        y1 = eval_x(x_plus)
                        if y1 is not None and y1 < y_best:
                            x_best = x_plus
                            y_best = y1
                            improved = True
                            no_improve_count = 0

                        y2 = eval_x(x_minus)
                        if y2 is not None and y2 < y_best:
                            x_best = x_minus
                            y_best = y2
                            improved = True
                            no_improve_count = 0

            # If heavily stagnating, diversify a bit (restart-like expansion)
            if no_improve_count >= max_no_improve:
                remaining = self.budget - evals
                if remaining <= 0:
                    break

                # Expand sigma and sample one or two fresh candidates to shake out.
                sigma = np.minimum(sigma * 1.7, sigma_max)

                # Evaluate a small diversification batch around a noisy center toward bounds
                # to escape local traps.
                shake_n = min(2, remaining)
                X_shake = []
                for _k in range(shake_n):
                    # Directionally bias towards random convex combinations of bounds
                    t = rng.random(self.dim)
                    center = lb * (1.0 - t) + ub * t
                    # Combine with best to keep promising area
                    alpha = rng.random()
                    c = (1.0 - alpha) * x_best + alpha * center
                    z = rng.standard_normal(self.dim)
                    X_shake.append(clip(c + z * sigma))

                for x_cand in X_shake:
                    if evals >= self.budget:
                        break
                    y = eval_x(x_cand)
                    if y is not None and y < y_best:
                        x_best = x_cand
                        y_best = y
                        no_improve_count = 0
                        sigma = np.maximum(sigma * 0.9, sigma_min)
                        break

        # Final safety: ensure returned types are numpy arrays and float
        return np.asarray(x_best, dtype=float), float(y_best)
