# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization strategy
# based on repeated local coordinate searches combined with a lightweight
# global restart mechanism. It works in any dimension and only relies on
# function evaluations, respecting the provided evaluation budget.
# Search state: The algorithm keeps a current best solution x_best/y_best,
# a step size (sigma) controlling perturbations, and an iteration counter that
# tracks consumed evaluations.
# Candidate generation: For each cycle, it samples several candidates by
# applying random perturbations to the current best. It also performs a
# deterministic coordinate sweep (± along a random permutation of axes)
# to sharpen local improvements. Candidates are clipped to the feasible bounds.
# Selection and replacement: Among the batch of evaluated candidates, the best
# (lowest objective value) replaces the current best. If no improvement occurs
# for several cycles, the algorithm triggers a restart.
# Adaptation: Step size sigma is decreased when improvements are found
# (focused search) and increased slightly when stagnation is detected
# (to escape local minima).
# Exploration mechanisms: Random perturbations and restarts provide exploration
# across the space.
# Exploitation mechanisms: The coordinate sweep and decreasing sigma around the
# best point encourage exploitation and fast local convergence.
# Boundary handling: Any candidate is projected back into the search box using
# clipping to the provided lower/upper bounds (or bounds.lb/bounds.ub).
# Budget strategy: The total number of objective evaluations is capped by the
# provided budget. The algorithm queries the objective only through a wrapper
# that decrements remaining evaluations and stops generating new candidates
# when the budget is exhausted.
# Closest known influences: Combines ideas similar to evolution strategies with
# coordinate descent refinement and restart-on-stagnation, tailored for
# small/medium evaluation budgets.
# Novelty or unusual aspects: Uses a hybrid batch of random direction probes
# plus a coordinate sweep each cycle, with conservative budget-aware batching.
# Failure modes: If the objective is very noisy or the budget is extremely low,
# the method may rely on insufficient sampling; in flat landscapes it may
# stagnate, but restarts and sigma adaptation help somewhat.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        d = self.dim
        if d <= 0:
            raise ValueError("dim must be positive")

        # Read bounds from func.lower/func.upper or func.bounds.lb/func.bounds.ub
        lb, ub = None, None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            lb = np.asarray(b.lb, dtype=float)
            ub = np.asarray(b.ub, dtype=float)
        else:
            raise AttributeError("Objective must provide bounds via lower/upper or bounds.lb/bounds.ub")

        if lb.shape == () or ub.shape == ():
            lb = np.full(d, float(lb), dtype=float)
            ub = np.full(d, float(ub), dtype=float)
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.size != d or ub.size != d:
            raise ValueError("Bounds shape does not match dim")
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            raise ValueError("Bounds must be finite")
        if np.any(ub < lb):
            raise ValueError("Upper bounds must be >= lower bounds")

        rng = np.random  # harness sets global seed via np.random.seed

        # Evaluation budget guard
        remaining = self.budget
        if remaining <= 0:
            # No evaluations possible; return a feasible point deterministically
            mid = (lb + ub) / 2.0
            return mid, float("inf")

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        # Objective evaluation wrapper
        def eval_x(x):
            nonlocal remaining
            if remaining <= 0:
                return float("inf")
            remaining -= 1
            x = np.asarray(x, dtype=float)
            y = func(x)
            # Ensure scalar float
            try:
                y = float(y)
            except Exception:
                y = float(np.asarray(y).reshape(()))
            return y

        # Initial point: mid + a small random jitter (if there is room)
        mid = (lb + ub) / 2.0
        span = ub - lb
        span[span == 0] = 1.0  # avoid zero range issues

        # Conservative initial sigma tied to bounds
        sigma = 0.3 * np.min(span) if np.min(span) > 0 else 0.1
        sigma = float(max(sigma, 1e-12))

        # Choose a feasible random start with some probability, otherwise midpoint.
        if rng.rand() < 0.5:
            x_best = clip(mid + rng.uniform(-0.1, 0.1, size=d) * span)
        else:
            x_best = clip(mid)
        y_best = eval_x(x_best)

        # Heuristic control parameters scaled with dimension and budget
        # cycles: number of main outer loops
        cycles = max(1, int(self.budget // (max(1, 2 * d + 4))))
        # Candidates per cycle (excluding coordinate sweep) - keep modest for budget safety
        rand_batch = int(max(4, min(12, self.budget // max(1, 2 * cycles) - (d + 2))))
        rand_batch = max(4, rand_batch)

        stagnation = 0
        stagnation_limit = max(3, int(0.15 * cycles) + 3)

        # Main search loop
        # Each cycle: evaluate random perturbations; then coordinate sweep; update sigma.
        for _ in range(cycles):
            if remaining <= 0:
                break

            improved = False
            best_local_x = x_best
            best_local_y = y_best

            # --- Exploration: random perturbations around current best ---
            # Sample Gaussian steps scaled by sigma and bounds span.
            # Add slight anisotropy using per-coordinate scaling to handle different ranges.
            if remaining > 0:
                evals_before = remaining
                # Evaluate until batch size or budget exhaustion
                n_rand = min(rand_batch, remaining)
                if n_rand > 0:
                    steps = rng.normal(0.0, 1.0, size=(n_rand, d))
                    # Scale steps by sigma and normalized span
                    span_scale = span / np.max(span) if np.max(span) > 0 else 1.0
                    # Center on x_best
                    X = clip(x_best + (sigma * span_scale) * steps)
                    for i in range(n_rand):
                        y = eval_x(X[i])
                        if y < best_local_y:
                            best_local_y = y
                            best_local_x = X[i]
                            improved = True

                # Adjust sigma based on whether we improved in this batch (with small gains)
                if remaining <= 0:
                    break

            # --- Exploitation: coordinate sweep (± along axes) ---
            # Try steps that decrease/increase around sigma; only do if budget allows.
            if remaining > 0:
                # Choose an axis order randomized each cycle
                axes = np.arange(d)
                rng.shuffle(axes)

                # Coordinate step sizes
                # Larger trial at start of cycle, smaller if sigma has shrunk.
                coord_step1 = sigma
                coord_step2 = sigma * 0.5

                # Limit coordinate trials by remaining evaluations:
                # For each axis, try at most 2 evaluations (±).
                max_axes = min(d, remaining // 2) if remaining > 0 else 0
                max_axes = max_axes if max_axes > 0 else 0

                for ai in axes[:max_axes]:
                    if remaining <= 0:
                        break
                    x_try = best_local_x.copy()
                    # ± coord_step1
                    for sgn, step in ((+1.0, coord_step1), (-1.0, coord_step1)):
                        if remaining <= 0:
                            break
                        x_try2 = x_try.copy()
                        x_try2[ai] = x_try2[ai] + sgn * step
                        x_try2 = clip(x_try2)
                        y = eval_x(x_try2)
                        if y < best_local_y:
                            best_local_y = y
                            best_local_x = x_try2
                            improved = True
                    if remaining <= 0:
                        break

                    # If improvement is found, do a smaller follow-up step on this axis
                    if improved and remaining > 0:
                        for sgn in (1.0, -1.0):
                            if remaining <= 0:
                                break
                            x_try3 = best_local_x.copy()
                            x_try3[ai] = x_try3[ai] + sgn * coord_step2
                            x_try3 = clip(x_try3)
                            y = eval_x(x_try3)
                            if y < best_local_y:
                                best_local_y = y
                                best_local_x = x_try3
                                improved = True

            # Update global best
            if best_local_y < y_best:
                x_best, y_best = best_local_x, best_local_y

            # --- Adapt sigma and manage stagnation ---
            if improved or best_local_y < y_best + 0.0:
                # If we found a strictly better point, tighten search
                sigma *= 0.85
                stagnation = 0
            else:
                stagnation += 1
                # Mildly widen search on stagnation
                sigma *= 1.08

            # Ensure sigma doesn't collapse to zero in bounded boxes
            sigma_min = 1e-12
            sigma = float(max(sigma, sigma_min))

            # --- Restart mechanism on stagnation ---
            if stagnation >= stagnation_limit and remaining > 0:
                # Jump to a random feasible point and reinitialize sigma.
                # Use remaining budget to evaluate a few random candidates to avoid
                # spending too much on restarts.
                n_restart = min(5, remaining)
                if n_restart > 0:
                    # Random points uniform in box
                    Xr = lb + rng.rand(n_restart, d) * (ub - lb)
                    best_r_y = float("inf")
                    best_r_x = None
                    for i in range(n_restart):
                        y = eval_x(Xr[i])
                        if y < best_r_y:
                            best_r_y = y
                            best_r_x = Xr[i]
                    if best_r_x is not None and best_r_y < y_best:
                        x_best, y_best = best_r_x, best_r_y
                    # Reset sigma based on box scale
                    sigma = 0.5 * np.min(span) if np.min(span) > 0 else 0.1
                    sigma = float(max(sigma, 1e-12))
                stagnation = 0

        return x_best, y_best
