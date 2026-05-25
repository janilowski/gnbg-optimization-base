import numpy as np

# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer using a
# randomized trust-region-like pattern search with occasional restarts.
# Search state: Maintains a current best solution x_best and its objective
# value y_best, plus a step-size sigma controlling candidate perturbations.
# Candidate generation: At each iteration, samples candidates around the
# current best by adding Gaussian perturbations scaled by sigma and a
# direction drawn from a mixture of random and coordinate-wise directions.
# Selection and replacement: Evaluates candidates (without exceeding budget),
# then moves to the best candidate if it improves y_best. The step-size is
# adapted based on whether improvements occur.
# Adaptation: If improvement happens, sigma decreases slightly slower (focus
# search); if no improvement, sigma decays more aggressively to intensify
# or later triggers a restart with sigma reset and a new random center.
# Exploration mechanisms: Random directions and Gaussian noise provide global
# exploration; periodic restarts encourage escaping local minima.
# Exploitation mechanisms: Smaller sigma and best-so-far tracking focus the
# search near promising regions; coordinate-perturbation candidates help
# in axis-aligned structures.
# Boundary handling: Candidates are clipped to the provided bounds (from
# func.lower/upper or func.bounds.lb/ub). If bounds are infinite, clipping is
# skipped safely.
# Budget strategy: Uses a strict evaluation counter; the number of function
# calls is capped by the provided budget. A small initial design is sampled
# from the bounds to establish a starting point.
# Closest known influences: Pattern search / evolution-strategy style
# self-adaptive step-size rules, simplified for standard library + numpy.
# Novelty or unusual aspects: Combines Gaussian direction sampling with a
# coordinate perturbation heuristic and uses a restart rule tied to a window
# of consecutive non-improvements.
# Failure modes: With extremely small budgets or highly irregular objectives,
# convergence may be limited; clipping can reduce effective exploration near
# bounds; restart frequency may be suboptimal for some landscapes.
# ALGORITHM_ANALYSIS_NOTE_END


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        if budget <= 0:
            # No evaluations allowed; return a deterministic point.
            x0 = np.zeros(dim, dtype=float)
            return x0, float("inf")

        lb, ub = self._read_bounds(func, dim)
        rng = np.random.default_rng()  # harness sets global seed; this RNG will still be deterministic only if seed set globally.

        def clip_to_bounds(x):
            if lb is None and ub is None:
                return x
            if lb is None:
                lb_arr = np.full(dim, -np.inf, dtype=float)
            else:
                lb_arr = lb
            if ub is None:
                ub_arr = np.full(dim, np.inf, dtype=float)
            else:
                ub_arr = ub
            return np.minimum(np.maximum(x, lb_arr), ub_arr)

        def scale_range():
            # Estimate a characteristic scale from bounds to choose sigma.
            if lb is None or ub is None:
                return 1.0
            span = ub - lb
            finite_span = span[np.isfinite(span)]
            if finite_span.size == 0:
                return 1.0
            med = np.median(finite_span)
            if med <= 0 or not np.isfinite(med):
                return 1.0
            return float(med)

        # Strict evaluation counter
        evals = 0

        def eval_obj(x):
            nonlocal evals
            if evals >= budget:
                return None
            x = np.asarray(x, dtype=float)
            y = func(x)
            evals += 1
            return float(y)

        # Determine initial center(s) from bounds
        if lb is None or ub is None:
            center = np.zeros(dim, dtype=float)
        else:
            # start from a random point in the box to add robustness
            center = lb + rng.random(dim) * (ub - lb)
        center = clip_to_bounds(center)

        # Initial sigma based on bounds span (or 1.0 if unavailable)
        base_scale = scale_range()
        sigma = 0.5 * base_scale
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        # Evaluate initial candidates: small "seed" set
        # Choose seed count based on budget; keep it small for budget-limited runs.
        seed_count = min(6, budget)
        # Always include center first
        seed_xs = [center]
        for _ in range(seed_count - 1):
            if lb is not None and ub is not None and np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
                x = lb + rng.random(dim) * (ub - lb)
            else:
                x = rng.normal(0.0, 1.0, size=dim) * base_scale + center
            seed_xs.append(clip_to_bounds(x))

        best_x = None
        best_y = float("inf")

        for x in seed_xs:
            if evals >= budget:
                break
            y = eval_obj(x)
            if y is None:
                break
            if y < best_y or best_x is None:
                best_y, best_x = y, np.array(x, copy=True)

        if best_x is None:
            best_x = center.copy()
            best_y = float("inf")

        # Parameters for the search dynamics
        # Candidate count per "iteration" (only if budget allows).
        k = 4 + int(np.sqrt(dim))  # scales mildly with dimension
        k = max(4, min(k, 12))
        # Multipliers for sigma adaptation
        improve_down = 0.82
        no_improve_down = 0.60
        increase_on_restart = 1.20
        min_sigma = 1e-12 * (base_scale if np.isfinite(base_scale) else 1.0)
        max_sigma = 2.0 * (base_scale if np.isfinite(base_scale) else 1.0)
        if not np.isfinite(max_sigma) or max_sigma <= 0:
            max_sigma = 1.0

        # Restart heuristic: after several stagnant attempts, re-center randomly.
        # Keep it budget-aware: allow more stagnation when budget is larger.
        stagnation_window = 12 + int(0.05 * budget)
        stagnation_window = max(12, min(stagnation_window, 50))
        no_improve_streak = 0

        # Main loop
        while evals < budget:
            remaining = budget - evals
            cand_count = min(k, remaining)

            candidates = []

            # Mixture: Gaussian perturbations + coordinate kicks + occasional purely random point
            p_random = 0.08 if lb is None or ub is None else 0.05
            p_coord = 0.35

            for i in range(cand_count):
                if lb is not None and ub is not None and rng.random() < p_random:
                    # Pure exploration: sample in the box
                    x = lb + rng.random(dim) * (ub - lb)
                    candidates.append(clip_to_bounds(x))
                    continue

                x = best_x.copy()

                # Choose a direction type
                if rng.random() < p_coord:
                    # Coordinate-wise perturbation: pick a few coordinates
                    x_new = x.copy()
                    m = 1 + int(rng.integers(0, max(2, dim // 4 + 1)))
                    idx = rng.choice(dim, size=m, replace=False)
                    # Slightly heavier tails for coordinate moves
                    step = sigma * rng.normal(0.0, 1.0, size=m)
                    x_new[idx] = x_new[idx] + step
                    candidates.append(clip_to_bounds(x_new))
                else:
                    # Gaussian perturbation around best
                    # Use anistropic diagonal scaling using random multipliers
                    # to better handle different coordinate scales.
                    diag = rng.lognormal(mean=0.0, sigma=0.35, size=dim)
                    z = rng.normal(0.0, 1.0, size=dim)
                    # With small probability, amplify one direction
                    if dim > 1 and rng.random() < 0.12:
                        j = int(rng.integers(0, dim))
                        diag[j] *= 3.0
                    x_new = x + (sigma * z * diag)
                    candidates.append(clip_to_bounds(x_new))

            # Evaluate and select best candidate
            local_best_y = best_y
            local_best_x = best_x

            for x in candidates:
                if evals >= budget:
                    break
                y = eval_obj(x)
                if y is None:
                    break
                if y < local_best_y:
                    local_best_y = y
                    local_best_x = np.array(x, copy=True)

            if local_best_y < best_y:
                # Improvement: accept and expand a little in search radius for robustness
                best_x = local_best_x
                best_y = local_best_y
                no_improve_streak = 0
                # Decrease sigma more conservatively to keep making progress
                sigma = max(min_sigma, sigma * improve_down)
            else:
                # No improvement: decrease sigma and count stagnation
                no_improve_streak += 1
                sigma = max(min_sigma, sigma * no_improve_down)

            if no_improve_streak >= stagnation_window and evals < budget:
                # Restart: re-center near a random point and reset sigma
                if lb is None or ub is None:
                    center = rng.normal(0.0, 1.0, size=dim) * base_scale
                else:
                    center = lb + rng.random(dim) * (ub - lb)
                center = clip_to_bounds(center)

                best_x = center
                # Evaluate new center to ensure best_y is consistent with objective evaluations
                y_center = eval_obj(best_x)
                if y_center is not None:
                    best_y = y_center
                # Reset sigma to encourage exploration then refinement
                sigma = min(max_sigma, (sigma * increase_on_restart) + 0.5 * base_scale)
                no_improve_streak = 0

            # Final safety: if sigma becomes too small but budget remains, gently re-energize
            if sigma <= min_sigma and evals < budget:
                sigma = min(max_sigma, base_scale if base_scale > 0 else 1.0)

        return np.array(best_x, dtype=float, copy=True), float(best_y)

    @staticmethod
    def _read_bounds(func, dim):
        # Prefer func.lower/func.upper
        lb = None
        ub = None

        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(getattr(func, "lower"), dtype=float).reshape(-1)
            ub = np.asarray(getattr(func, "upper"), dtype=float).reshape(-1)
        elif hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            # Support b.lb / b.ub
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(getattr(b, "lb"), dtype=float).reshape(-1)
                ub = np.asarray(getattr(b, "ub"), dtype=float).reshape(-1)

        if lb is not None and ub is not None:
            if lb.size != dim or ub.size != dim:
                # If bounds sizes mismatch, fall back to no clipping
                return None, None
            return lb, ub

        return None, None
