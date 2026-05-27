# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization algorithm (derivative-free)
# using a mixture of coordinate-wise exploration and a simplex-like centroid search.
# It maintains a small population of candidate points and repeatedly perturbs the current
# best using a gradually shrinking step size while also injecting occasional random moves
# to escape stagnation.
# Search state: Keeps a best-so-far point (x_best, y_best), a small set of
# recent samples (population) and an evaluation counter to enforce the budget.
# Candidate generation: Proposes new candidates by (1) adding random Gaussian noise
# scaled by a dynamic step size to the best, (2) performing coordinate-wise
# one-dimensional probes around the best, and (3) generating simplex-inspired
# points around the centroid of the best candidates. Each proposed point is clipped
# to the feasible bounds.
# Selection and replacement: Uses strict minimization. If a candidate improves the
# best objective value, it replaces x_best. Population is refreshed around improved
# points to keep search focused.
# Adaptation: Step size starts proportional to the domain diameter and shrinks
# when no improvement is observed. It also expands slightly when progress stalls to
# encourage exploration.
# Exploration mechanisms: Random perturbations and occasional full reinitializations
# of a few candidates from the uniform distribution within bounds.
# Exploitation mechanisms: Coordinate-wise probes and centroid/simplex-like sampling
# biased toward the current best region.
# Boundary handling: After every candidate creation, values are clipped to [lb, ub].
# Budget strategy: Uses an internal evaluation counter and never calls the objective
# more times than the provided budget. The algorithm decides how many proposals it can
# afford based on remaining evaluations.
# Closest known influences: Related in spirit to evolution strategies and pattern search,
# with a simplex/centroid component and coordinate-wise local refinement.
# Novelty or unusual aspects: Combines a small-population centroid step with dynamic
# step-size adaptation and coordinate probing, all while staying within a strict evaluation
# budget and using only standard library + numpy.
# Failure modes: If the function is highly discontinuous or very noisy, strict
# improvement checks may cause premature shrinking; the occasional random injections and
# budget-limited restarts mitigate this. In extreme dimension, the coordinate probing
# may be less efficient, but it still provides some structured local search.
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
            raise ValueError("budget must be positive")

        # Read bounds robustly from either func.lower/func.upper or func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError("Objective must provide bounds via func.lower/func.upper or func.bounds.lb/ub")

        if lb.shape == ():
            lb = np.full(dim, float(lb))
        if ub.shape == ():
            ub = np.full(dim, float(ub))

        lb = lb.reshape(-1).astype(float)
        ub = ub.reshape(-1).astype(float)
        if lb.size != dim or ub.size != dim:
            raise ValueError("Bounds dimensionality does not match dim")

        # Ensure numerical ordering.
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        lb, ub = lo, hi

        rng = np.random

        evals = 0

        def clamp(x):
            return np.minimum(ub, np.maximum(lb, x))

        # Safe evaluation wrapper to never exceed budget.
        def eval_one(x):
            nonlocal evals
            if evals >= budget:
                # In case of unexpected extra calls, return a very bad value.
                return np.inf
            y = float(func(x))
            evals += 1
            return y

        # Domain diameter for initial step.
        diameter = float(np.linalg.norm(ub - lb))
        if not np.isfinite(diameter) or diameter <= 0:
            # Degenerate domain; best is any point.
            x0 = np.clip((lb + ub) / 2.0, lb, ub)
            y0 = eval_one(x0)
            return x0, y0

        # Initialize: sample a few points uniformly.
        # Choose population size based on budget/dimension (small and budget-aware).
        pop_size = int(np.clip(6 + dim // 3, 6, 24))
        pop_size = min(pop_size, budget)

        # Uniform initialization within bounds.
        X = lb + (ub - lb) * rng.rand(pop_size, dim)
        y = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            y[i] = eval_one(X[i])

        best_idx = int(np.argmin(y))
        x_best = X[best_idx].copy()
        y_best = float(y[best_idx])

        # Dynamic step size:
        # Start with a fraction of the diameter; shrink over time.
        step = (0.3 * diameter) / max(1.0, np.sqrt(dim))
        step = max(step, 1e-12)
        no_improve = 0

        # Precompute for candidate generation.
        coords = np.arange(dim)

        # We'll run until budget is reached.
        # Each loop proposes up to `batch` points.
        while evals < budget:
            remaining = budget - evals
            # Small batch size to keep budget tight.
            batch = int(min(remaining, 6 + dim // 4))

            # Keep a couple of top candidates for centroid sampling.
            # (Selection and replacement)
            top_k = int(min(len(X), max(3, batch // 2, dim)))
            top_idx = np.argsort(y)[:top_k]
            bests = X[top_idx]
            centroid = np.mean(bests, axis=0)

            new_points = []

            # Decide how much exploration vs exploitation to do.
            # If stagnating, explore more.
            explore_prob = 0.25 + 0.35 * (no_improve > 2)

            # Candidate 1: random Gaussian perturbations around best.
            n_gauss = int(batch * (0.5 if rng.rand() > explore_prob else 0.7))
            for _ in range(n_gauss):
                direction = rng.normal(size=dim)
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction /= direction_norm
                scale = step * (0.6 + 1.4 * rng.rand())
                x = x_best + scale * direction + 0.1 * step * rng.normal(size=dim)
                new_points.append(clamp(x))

            # Candidate 2: coordinate-wise one-dimensional probes (structured exploitation).
            n_coord = batch - len(new_points)
            if n_coord > 0:
                # Probe around best in a subset of coordinates.
                # Use budget-aware number of coordinate probes.
                n_probes = min(n_coord, dim, max(1, batch // 2))
                probe_coords = rng.choice(coords, size=n_probes, replace=False)
                for c in probe_coords:
                    for sign in (-1.0, 1.0):
                        if len(new_points) >= batch:
                            break
                        delta = step * (0.5 + 0.9 * rng.rand())
                        x = x_best.copy()
                        x[c] = x[c] + sign * delta
                        new_points.append(clamp(x))
                        if len(new_points) >= batch:
                            break
                # If still short, fill with centroid perturbations.
                while len(new_points) < batch and evals + len(new_points) < budget:
                    dir2 = rng.normal(size=dim)
                    x = centroid + step * (0.2 + 0.8 * rng.rand()) * dir2 / max(1e-12, np.linalg.norm(dir2))
                    new_points.append(clamp(x))

            # Candidate 3: simplex-like centroid step (extra exploitation)
            # Occasionally use it to steer toward centroid/bests.
            if rng.rand() < 0.35 and evals + len(new_points) <= budget:
                # Reflection of best across centroid: x_ref = centroid + alpha*(centroid - best)
                alpha = 1.0 + 1.5 * rng.rand()
                x_ref = centroid + alpha * (centroid - x_best)
                new_points.append(clamp(x_ref))

            # If we overshot batch due to simplex add, trim.
            if len(new_points) > batch:
                new_points = new_points[:batch]

            # Optional exploration injection: reinitialize a couple candidates uniformly.
            if rng.rand() < 0.2 and evals < budget:
                n_inj = min(2, max(0, batch // 4))
                for _ in range(n_inj):
                    if len(new_points) >= batch:
                        break
                    x = lb + (ub - lb) * rng.rand(dim)
                    new_points.append(x)

            # Evaluate and update.
            improved = False
            for x in new_points:
                if evals >= budget:
                    break
                yx = eval_one(x)
                if yx < y_best:
                    y_best = yx
                    x_best = x.copy()
                    improved = True
                    no_improve = 0
                else:
                    no_improve += 0  # keep stable; actual streak updated after batch

                # Update population by inserting and keeping size manageable.
                # Replace worst point with this candidate.
                # This keeps population concentrated without growing memory.
                if len(X) < pop_size:
                    X = np.vstack([X, x[None, :]])
                    y = np.append(y, yx)
                else:
                    worst = int(np.argmax(y))
                    if yx < y[worst]:
                        X[worst] = x
                        y[worst] = yx

            if not improved:
                no_improve += 1
                # Shrink step when no improvement.
                step *= 0.85
            else:
                # Mild shrink on improvement too, but less aggressive.
                step *= 0.95

            # If heavily stuck, slightly expand and inject random points around best.
            if no_improve >= 4:
                step *= 1.12
                no_improve = 0  # reset after expansion

            # Safety floor for step (avoid underflow stall too hard).
            if step < 1e-14 * (diameter / max(1.0, np.sqrt(dim))):
                step = 1e-14 * (diameter / max(1.0, np.sqrt(dim)))

        return x_best.copy(), float(y_best)
