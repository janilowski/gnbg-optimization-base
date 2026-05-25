# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer based on
# an adaptive coordinate/axis search with periodic isotropic exploration.
# Search state: Maintains a current best point x_best and its function value
# f_best, along with per-dimension step sizes (sigmas) that shrink or grow
# depending on success.
# Candidate generation: Proposes candidates by moving along individual
# coordinate directions (±sigma_i) and, occasionally, sampling an isotropic
# random Gaussian direction scaled by the current global step size.
# Selection and replacement: Evaluates candidates (never exceeding the budget),
# then greedily replaces the current best if a candidate improves it.
# Adaptation: If an axis move improves, the corresponding step size increases
# modestly; if it fails, the step size decreases (with a lower bound).
# Exploration mechanisms: Every few iterations (or when progress stalls),
# uses random Gaussian perturbations to escape local minima.
# Exploitation mechanisms: Repeated axis searches around the current best
# aggressively reduce step sizes to refine solutions.
# Boundary handling: Uses reflection at bounds to keep candidates feasible
# without biasing too strongly toward the interior.
# Budget strategy: Converts the given budget into a maximum number of function
# evaluations; stops early if the budget is exhausted.
# Closest known influences: Combines ideas from coordinate descent, step-size
# control in evolution strategies, and occasional random-restart style
# exploration—kept deliberately simple to be robust in many dimensions.
# Novelty or unusual aspects: Uses a per-dimension adaptive step size while also
# scheduling sparse isotropic exploration, balancing fast local refinement with
# budget-aware global probing.
# Failure modes: If the objective is extremely noisy or has very narrow
# feasible/meaningful regions, step sizes might shrink prematurely; reflection
# boundary handling may slow progress near hard walls.
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

        # --- Read bounds robustly ---
        lb, ub = None, None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
        if lb is None or ub is None:
            raise AttributeError("func must provide bounds via lower/upper or bounds.lb/bounds.ub")
        lb = np.broadcast_to(lb, (d,)).astype(float)
        ub = np.broadcast_to(ub, (d,)).astype(float)
        if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
            raise ValueError("Bounds must be finite")
        if np.any(ub <= lb):
            raise ValueError("Each upper bound must be strictly greater than lower bound")

        # --- Budget handling ---
        # We must not exceed the provided evaluation budget.
        budget = self.budget
        if budget <= 0:
            # With zero budget, we cannot evaluate. Return a deterministic in-bounds point.
            mid = 0.5 * (lb + ub)
            return np.array(mid, copy=True), float("inf")

        evals = 0

        def reflect_to_bounds(x):
            # Reflect x into [lb, ub] per coordinate.
            # Works with vectorized bounds; assumes lb < ub.
            x = np.asarray(x, dtype=float)
            lo = lb
            hi = ub
            span = hi - lo
            # Map into [0, span] using modulo, then reflect if in second half.
            # This avoids while-loops and handles large steps.
            y = (x - lo) % (2.0 * span)
            y = np.where(y <= span, y, 2.0 * span - y)
            return lo + y

        # Starting point: random uniform in bounds (harness sets numpy seed).
        x_best = reflect_to_bounds(np.random.uniform(lb, ub))
        f_best = func(x_best)
        evals += 1
        if evals >= budget:
            return np.array(x_best, copy=True), float(f_best)

        # Initial step sizes: fraction of domain width.
        width = ub - lb
        # Global scale: based on median width; robust across dimensions.
        medw = float(np.median(width))
        if medw <= 0:
            medw = 1.0
        global_step = 0.2 * medw
        sigmas = np.full(d, 0.2, dtype=float) * width  # per-dim relative steps
        # Ensure non-trivial sigmas
        min_sigma = 1e-12 * width
        sigmas = np.maximum(sigmas, min_sigma)

        # Small helpers
        it = 0
        # Stagnation counter to trigger occasional exploration
        no_improve = 0

        # Exploration schedule parameters
        explore_every = max(1, min(25, budget // max(1, d) + 1))
        global_shrink_on_fail = 0.8
        global_grow_on_success = 1.1

        # Greedy coordinate/axis search with adaptive step sizes.
        # Each iteration proposes up to 2*d axis moves, but we respect budget.
        while evals < budget:
            it += 1
            improved_in_round = False

            # Permute coordinate order to reduce bias across dimensions.
            coords = np.arange(d)
            np.random.shuffle(coords)

            # Decide if we should perform isotropic exploration this round.
            do_explore = (it % explore_every == 0) or (no_improve >= 5)

            for i in coords:
                if evals >= budget:
                    break

                # Try + direction
                step = sigmas[i]
                x_try = x_best.copy()
                x_try[i] = x_try[i] + step
                x_try = reflect_to_bounds(x_try)
                f_try = func(x_try)
                evals += 1
                if f_try < f_best:
                    x_best, f_best = x_try, f_try
                    # If success, enlarge step for this coordinate
                    sigmas[i] = sigmas[i] * global_grow_on_success
                    improved_in_round = True
                    no_improve = 0
                    if evals >= budget:
                        break
                else:
                    # If fail, shrink
                    sigmas[i] = max(sigmas[i] * global_shrink_on_fail, min_sigma[i])

                if evals >= budget:
                    break

                # Try - direction
                x_try = x_best.copy()
                x_try[i] = x_try[i] - step
                x_try = reflect_to_bounds(x_try)
                f_try = func(x_try)
                evals += 1
                if f_try < f_best:
                    x_best, f_best = x_try, f_try
                    sigmas[i] = sigmas[i] * global_grow_on_success
                    improved_in_round = True
                    no_improve = 0
                    if evals >= budget:
                        break
                else:
                    sigmas[i] = max(sigmas[i] * global_shrink_on_fail, min_sigma[i])

            # Optional isotropic exploration (random direction)
            if evals < budget and do_explore:
                # Use a few random probes but keep it budget-aware.
                # The number depends on remaining budget and dimension.
                remaining = budget - evals
                # Probe count capped to keep runtime predictable.
                k = 1
                if d <= 10:
                    k = min(3, remaining)
                else:
                    k = min(2, remaining)

                # Isotropic step magnitude tied to the spread.
                # Use current global_step and also shrink it gradually with progress.
                # global_step shrinks on no improvement.
                if not improved_in_round:
                    global_step *= 0.9
                else:
                    global_step *= 1.02
                global_step = max(global_step, 1e-12 * medw)

                for _ in range(k):
                    if evals >= budget:
                        break
                    # Random direction: standard normal then normalized.
                    v = np.random.normal(size=d)
                    nv = np.linalg.norm(v)
                    if nv == 0:
                        continue
                    v = v / nv
                    # Scale by global_step and optionally per-dim spread.
                    # Use width to respect anisotropic domains.
                    scale = global_step
                    # Slightly bias towards dimensions with larger current sigmas.
                    # (This helps when some axes already refined.)
                    anis = np.sqrt(np.maximum(sigmas, min_sigma))
                    anis = anis / (np.linalg.norm(anis) / np.sqrt(d))
                    step_vec = v * scale * (anis / (np.mean(anis) + 1e-18))
                    x_try = reflect_to_bounds(x_best + step_vec)
                    f_try = func(x_try)
                    evals += 1
                    if f_try < f_best:
                        x_best, f_best = x_try, f_try
                        improved_in_round = True
                        no_improve = 0

            if not improved_in_round:
                no_improve += 1
                # If we're not improving, contract steps a bit globally.
                sigmas = np.maximum(sigmas * 0.95, min_sigma)
                global_step = max(global_step * 0.95, 1e-12 * medw)
            else:
                # If improving, slightly expand global step for potential discovery.
                global_step = max(global_step * 1.02, 1e-12 * medw)

        return np.array(x_best, copy=True), float(f_best)
