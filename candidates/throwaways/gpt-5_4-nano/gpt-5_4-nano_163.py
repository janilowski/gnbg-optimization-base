# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization strategy
# based on a derivative-free coordinate search with occasional random
# restarts. It maintains one incumbent solution and a step size that
# adapts based on whether improving moves are found within a local probe.
# Search state: The algorithm keeps track of the current best point (x_best),
# its objective value (y_best), a current step scale (step), and an evaluation
# counter (evals). It also keeps a few internal parameters controlling how
# aggressively it probes the neighborhood and how often it restarts.
# Candidate generation: Each iteration proposes candidates by sampling along
# coordinate-aligned directions (±step * e_i for selected indices), and
# also samples a small number of isotropic random points around the incumbent.
# Selection and replacement: Among all evaluated candidates in the current
# iteration, the best one replaces the incumbent if it improves y_best.
# Adaptation: If an improvement is found, step size is reduced (or kept
# moderate) to refine; if not, step size is increased to escape local minima.
# Exploration mechanisms: Random local perturbations plus occasional full random
# restarts (re-seeding around the bounds) provide global exploration.
# Exploitation mechanisms: Coordinate probes around the incumbent efficiently
# exploit smoothness or separability by searching along each dimension.
# Boundary handling: Candidates are clipped to the provided bounds after
# generation to ensure feasibility. If bounds are extremely tight, the search
# gracefully converges to that constrained region.
# Budget strategy: The algorithm strictly respects the provided evaluation
# budget by computing how many candidates it can still evaluate and by
# terminating immediately when the budget is exhausted. It always returns the
# best point found so far.
# Closest known influences: The design is influenced by common derivative-free
# heuristics such as coordinate pattern search, adaptive step-size schemes,
# and restart-based strategies for black-box optimization.
# Novelty or unusual aspects: The implementation uses a hybrid "coordinate
# pattern + isotropic perturbation" probe per iteration, and dynamically
# selects which dimensions to probe based on remaining budget and dimensionality.
# Failure modes: With very noisy or highly non-smooth objectives, step adaptation
# may bounce. In extremely high dimensions with extremely small budgets, the
# coordinate probing may cover only a tiny fraction of the space; restarts
# mitigate this but cannot guarantee global optimality.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Read bounds from func.lower/upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError(
                "func must provide either (lower, upper) or func.bounds.lb/func.bounds.ub"
            )

        if lb.shape != (dim,) or ub.shape != (dim,):
            lb = np.broadcast_to(lb, (dim,)).astype(float, copy=False)
            ub = np.broadcast_to(ub, (dim,)).astype(float, copy=False)

        # Ensure valid bounds ordering
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)

        # Small epsilon to avoid zero range problems
        span = hi - lo
        span_safe = np.where(span > 0, span, 1.0)
        tight = span <= 0

        def clip(x):
            return np.minimum(np.maximum(x, lo), hi)

        evals = 0
        x_best = None
        y_best = None

        # Initial step size: fraction of typical span
        # If bounds are tight, step becomes tiny.
        step = 0.25 * np.mean(span_safe)
        step = max(step, 1e-12)

        def eval_at(x):
            nonlocal evals, x_best, y_best
            x = clip(np.asarray(x, dtype=float))
            y = float(func(x))
            evals += 1
            if x_best is None or y < y_best:
                x_best = x
                y_best = y
            return y

        # If budget is too small to evaluate multiple points, just do one call.
        # We'll still return best found so far.
        if budget <= 0:
            raise ValueError("budget must be positive")

        # Start from a feasible random point (within bounds)
        # If all dimensions are tight, the point is deterministic.
        if np.all(tight):
            x0 = lo.copy()
        else:
            u = np.random.rand(dim)
            x0 = lo + u * (hi - lo)
            x0 = clip(x0)
        eval_at(x0)

        # Probe parameters
        # Each iteration will evaluate a "set" of candidates; we adapt how many to fit budget.
        # Coordinate probes: choose subset of dimensions to probe each iteration.
        coord_base = max(1, min(dim, 8))  # cap to keep per-iteration cost manageable
        isotropic_base = 3  # number of random perturbations per iteration

        # Main loop: strictly stop when budget is reached
        # We treat "iteration" as a group of candidate evaluations.
        while evals < budget:
            remaining = budget - evals

            # Decide how many coordinate directions we can afford
            # Candidates per iteration: 2*k (± for k coords) + m (isotropic)
            # Always include incumbent evaluation is not needed; only new candidates.
            k = min(dim, coord_base)
            # If remaining is tiny, reduce k and m.
            m = isotropic_base
            # Ensure we don't overshoot: 2k + m <= remaining
            # Also keep at least 2 probes if possible.
            max_k_by_budget = (remaining - m) // 2 if remaining > m else 0
            k = int(min(k, max(0, max_k_by_budget)))
            if remaining >= 2 and k == 0:
                # If we couldn't afford isotropic probes, try coordinate-only
                # Reduce m and recompute
                m = 0
                k = min(dim, coord_base, remaining // 2)

            # If still can't afford coordinate probes, take one isotropic step if possible.
            if k == 0 and m == 0:
                if remaining > 0:
                    # One final random candidate
                    if np.all(tight):
                        break
                    z = (np.random.randn(dim) * (0.1 * step)) / np.sqrt(dim)
                    x_cand = x_best + z
                    eval_at(x_cand)
                break

            candidates = []

            # --- Candidate generation: coordinate probes around incumbent ---
            # Select coordinates: mix sequential and random to cover more axes over time.
            if k > 0:
                # Dim tight handling: probing a tight dimension yields same value after clip.
                # We still include it, but it's harmless; selection tries to avoid wasted probes.
                non_tight = np.flatnonzero(~tight)
                if non_tight.size == 0:
                    k_eff = 0
                else:
                    # Choose k coordinates from non-tight dims if possible
                    k_eff = min(k, non_tight.size)
                    if non_tight.size <= k_eff:
                        idx = non_tight
                    else:
                        # Random subset
                        idx = np.random.choice(non_tight, size=k_eff, replace=False)
                if k > 0 and non_tight.size > 0:
                    # Propose ± along chosen coordinates
                    for i in idx:
                        # If bound span is tiny, skip by scaling step down (still safe)
                        di = step
                        if span[i] > 0:
                            di = step * (span[i] / span_safe.mean())
                        xi_plus = x_best.copy()
                        xi_minus = x_best.copy()
                        xi_plus[i] = xi_plus[i] + di
                        xi_minus[i] = xi_minus[i] - di
                        candidates.append(xi_plus)
                        if len(candidates) < remaining:
                            candidates.append(xi_minus)

            # --- Candidate generation: isotropic random perturbations ---
            if len(candidates) < remaining and m > 0 and not np.all(tight):
                # m points around incumbent with decreasing/increasing magnitude
                # Normalize noise magnitude so it is roughly comparable across dim.
                base_sigma = step / max(1.0, np.sqrt(dim))
                for _ in range(m):
                    if len(candidates) >= remaining:
                        break
                    noise = np.random.randn(dim) * base_sigma
                    # Slightly bias with a random direction
                    x_cand = x_best + noise
                    candidates.append(x_cand)

            # Evaluate candidates (strictly bounded by remaining)
            improved = False
            best_iter_y = y_best
            # Evaluate in random order to reduce systematic bias
            if len(candidates) > 1:
                order = np.random.permutation(len(candidates))
                candidates = [candidates[i] for i in order]

            for x in candidates:
                if evals >= budget:
                    break
                y = float(func(clip(np.asarray(x, dtype=float))))
                # Manually update best and evals to avoid double clip/eval wrapper overhead
                evals += 1
                if y < y_best:
                    x_best = clip(np.asarray(x, dtype=float))
                    y_best = y
                    improved = True
                    best_iter_y = y

            # --- Adaptation: step size update ---
            # If improvement, reduce step to exploit; otherwise increase to explore.
            if improved:
                # Mild contraction to refine
                step *= 0.8
            else:
                # Mild expansion; also handle extremely tight bounds by not exploding
                step *= 1.25

            # Guard step size to reasonable range relative to bounds
            # Upper bound: span_safe.mean() (or larger if span_safe is large)
            upper = max(1e-12, 0.5 * np.mean(span_safe))
            lower = 1e-12
            if step > upper:
                step = upper
            if step < lower:
                step = lower

            # --- Exploration mechanism: occasional restart ---
            # If no progress for a while or step is very small, restart from a random point.
            # Use objective-independent trigger to avoid needing history storage.
            # We approximate this by restarting when step has become too small or
            # when we are near budget with poor improvement.
            if (not improved and step <= 2e-12) or (np.random.rand() < 0.03 and not improved and remaining < max(10, dim)):
                if evals < budget and not np.all(tight):
                    u = np.random.rand(dim)
                    x_restart = lo + u * (hi - lo)
                    eval_at(x_restart)
                    step = max(0.25 * np.mean(span_safe), 1e-12)

        return x_best, y_best
