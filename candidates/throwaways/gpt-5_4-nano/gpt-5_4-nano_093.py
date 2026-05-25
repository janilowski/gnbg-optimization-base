# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, gradient-free black-box minimization algorithm
# that mixes local quadratic/coordinate-free search with global random restarts.
# It maintains a small population of candidate points and iteratively improves
# the best-so-far solution while controlling step sizes.
#
# Search state: Keeps a current best point (x_best, y_best), a set of
# population points (X, Y), and a single global step-size (sigma) that shrinks
# when improvements are found and expands slightly otherwise.
#
# Candidate generation: Each iteration generates candidates around the current
# best using (1) Gaussian perturbations in all coordinates, (2) coordinate
# perturbations (a random subset of axes), and (3) occasional uniform samples
# across the search box (to avoid stagnation). Each candidate is clipped into
# bounds.
#
# Selection and replacement: Candidates are evaluated, and the population is
# replaced by the best points among the union of old population and new
# candidates. The global best is updated if any candidate improves it.
#
# Adaptation: sigma is adapted using a simple success rule based on whether
# the best in the newly evaluated set improved over the current best.
#
# Exploration mechanisms: Periodic random restarts via uniform sampling and
# forced wider perturbations when no progress is seen for a while.
#
# Exploitation mechanisms: When progress occurs, sigma shrinks and the search
# intensifies around the current best; local refinements use smaller,
# directionally diverse perturbations.
#
# Boundary handling: Every candidate point is clipped to the provided bounds
# (lower/upper) before evaluation. This guarantees feasible evaluations.
#
# Budget strategy: Tracks the evaluation count explicitly and never exceeds the
# provided budget. The algorithm decides how many points to evaluate per
# iteration based on remaining budget.
#
# Closest known influences: Inspired by evolutionary strategies (ES) / CMA-like
# behavior, but kept minimal: uses a best-centered sampling, population
# selection, and step-size adaptation (success-based).
#
# Novelty or unusual aspects: Combines best-centered Gaussian sampling with
# randomized coordinate perturbations and a lightweight restart trigger,
# all implemented without any external dependencies beyond numpy.
#
# Failure modes: In very high dimensions or extremely ill-conditioned
# landscapes, the simple isotropic step-size adaptation may converge slowly.
# If bounds are very tight or the objective is noisy/erratic, progress may be
# limited and sigma adaptation could oscillate.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # ---- Read bounds from func ----
        lb = None
        ub = None
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)

        if lb is None or ub is None:
            raise AttributeError("Function must provide bounds via func.lower/func.upper or func.bounds.lb/ub.")
        if lb.shape != (dim,) or ub.shape != (dim,):
            # Be forgiving: allow bounds to broadcast/reshape to dim
            lb = np.asarray(lb, dtype=float).reshape(-1)
            ub = np.asarray(ub, dtype=float).reshape(-1)
            if lb.size != dim or ub.size != dim:
                raise ValueError(f"Bounds must have size dim={dim}. Got lb={lb.size}, ub={ub.size}.")

        lower = np.minimum(lb, ub)
        upper = np.maximum(lb, ub)
        width = upper - lower
        # Guard against zero-width dimensions
        safe_width = np.where(width > 0, width, 1.0)

        evals = 0
        best_x = None
        best_y = None

        def clip(x):
            return np.minimum(upper, np.maximum(lower, x))

        def eval_point(x):
            nonlocal evals, best_x, best_y
            y = func(np.asarray(x, dtype=float))
            evals += 1
            # Initialize/update best
            if best_y is None or y < best_y:
                best_y = float(y)
                best_x = np.asarray(x, dtype=float).copy()
            return float(y)

        # ---- Budget guard ----
        if budget <= 0:
            # No evaluation allowed; return any feasible point
            x0 = clip(np.zeros(dim, dtype=float))
            return x0, float(func(x0)) if False else (x0, float("inf"))

        # Initial candidate(s)
        # Start at center + a few random points (uses as much of budget as possible but stays compact).
        center = lower + 0.5 * width
        sigma0 = 0.5 * np.linalg.norm(safe_width) / np.sqrt(dim) if dim > 0 else 1.0
        sigma = max(1e-12, float(sigma0))

        # Choose an initial population size proportional to dim, capped.
        pop_size = int(min(16, max(2, 2 + dim // 2)))
        pop_size = min(pop_size, budget)

        # Create initial population, always feasible
        X = np.empty((pop_size, dim), dtype=float)
        Y = np.empty(pop_size, dtype=float)

        # Always evaluate at least one point: center and randoms
        n_init = min(pop_size, budget)
        # Ensure center evaluation included when possible
        k = 0
        if n_init > 0:
            x = clip(center.copy())
            X[k] = x
            Y[k] = eval_point(x)
            k += 1

        # Remaining init points uniformly random
        while k < n_init:
            x = lower + np.random.rand(dim) * (upper - lower)
            X[k] = x
            Y[k] = eval_point(x)
            k += 1

        # Sort and set best_x/y from arrays (already tracked, but ensure consistency)
        order = np.argsort(Y)
        X = X[order]
        Y = Y[order]
        if best_y is None:
            best_x = X[0].copy()
            best_y = float(Y[0])

        # If budget permits, pad population to pop_size (for selection diversity)
        while k < pop_size and evals < budget:
            x = lower + np.random.rand(dim) * (upper - lower)
            X[k] = x
            Y[k] = eval_point(x)
            k += 1
        if evals < budget:
            order = np.argsort(Y[:k])
            X[:k] = X[:k][order]
            Y[:k] = Y[:k][order]
            X = X[:k]
            Y = Y[:k]
            pop_size = X.shape[0]

        # Iteration parameters
        stagnation = 0
        best_global = best_y

        # Candidate batch size per iteration
        # Keep it small so we can react to the budget; also at least 1.
        base_batch = int(min(12, max(4, pop_size)))
        max_batch = int(min(24, max(4, 2 * pop_size)))
        # How often to do more exploratory sampling
        restart_patience = int(max(6, 2 + dim // 4))

        # Helper: generate candidates around best_x
        def generate_candidates(num):
            nonlocal sigma, stagnation
            # Mix three mechanisms: Gaussian (dense), coordinate (sparse), and uniform (restart-like).
            # Allocation depends on stagnation.
            # During stagnation, increase uniform exploration.
            if width.max() <= 0:
                # Degenerate bounds: only one feasible point.
                return np.tile(best_x, (num, 1))

            # Fraction of uniform exploration
            frac_uniform = 0.05
            if stagnation > 0:
                frac_uniform = min(0.55, frac_uniform + 0.05 * min(stagnation, 10))
            n_uniform = int(round(frac_uniform * num))
            n_uniform = min(n_uniform, num)

            n_gauss = int(round((num - n_uniform) * 0.7))
            n_coord = (num - n_uniform) - n_gauss

            candidates = np.empty((num, dim), dtype=float)
            idx = 0

            # Uniform candidates across box
            for _ in range(n_uniform):
                x = lower + np.random.rand(dim) * (upper - lower)
                candidates[idx] = x
                idx += 1

            # Gaussian candidates around best_x
            # Use isotropic sigma scaled by box width to be scale-aware.
            scale = safe_width / np.sqrt(dim)
            # slightly widen early if sigma is small relative to box
            gauss_scale = float(sigma) / (np.linalg.norm(scale) + 1e-12)
            if not np.isfinite(gauss_scale) or gauss_scale <= 0:
                gauss_scale = 1.0
            for _ in range(n_gauss):
                z = np.random.randn(dim)
                x = best_x + gauss_scale * (scale * z)
                candidates[idx] = x
                idx += 1

            # Coordinate perturbations (pick a random subset of axes)
            # Step sizes for coordinate search are scaled by sigma and coordinate widths.
            # This helps in axis-aligned or separable landscapes.
            for _ in range(n_coord):
                x = best_x.copy()
                # Choose ~10% of axes, at least 1, at most dim
                m = int(max(1, min(dim, 1 + dim // 10)))
                axes = np.random.choice(dim, size=m, replace=False)
                step = (float(sigma) / np.sqrt(dim)) * np.random.randn()
                # Each selected axis gets an independent random sign/step, scaled by width
                signs = np.where(np.random.rand(m) < 0.5, -1.0, 1.0)
                x[axes] = x[axes] + signs * step * (safe_width[axes] / (np.mean(safe_width) + 1e-12))
                candidates[idx] = x
                idx += 1

            # If rounding caused mismatch, trim/pad
            if idx < num:
                candidates[idx:] = best_x
            elif idx > num:
                candidates = candidates[:num]
            return candidates

        # Main loop
        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break

            # Adjust batch size to remaining budget
            batch = int(min(max_batch, base_batch, remaining))
            # Ensure at least 1 eval
            if batch < 1:
                break

            # Generate candidates and clip
            C = generate_candidates(batch)
            C = clip(C)

            # Evaluate candidates
            Ys = np.empty(batch, dtype=float)
            for i in range(batch):
                if evals >= budget:
                    Ys = Ys[:i]
                    C = C[:i]
                    break
                Ys[i] = eval_point(C[i])

            # Selection/replacement: keep best pop_size from union
            # Combine current population with new candidates
            if C.shape[0] == 0:
                break

            # Current pop may be smaller than planned; use its size
            cur_n = X.shape[0]
            # Build union arrays
            U_n = cur_n + C.shape[0]
            # Guard for allocation
            U_X = np.empty((U_n, dim), dtype=float)
            U_Y = np.empty(U_n, dtype=float)
            U_X[:cur_n] = X
            U_Y[:cur_n] = Y
            U_X[cur_n:] = C
            U_Y[cur_n:] = Ys

            ord_u = np.argsort(U_Y)
            keep = min(pop_size, U_n)
            ord_u = ord_u[:keep]
            X = U_X[ord_u]
            Y = U_Y[ord_u]

            # Update best and adaptation
            new_best = float(np.min(Y))
            improved = new_best < best_global - 1e-15
            if improved:
                best_global = new_best
                # Success: shrink sigma moderately to exploit
                stagnation = 0
                sigma *= 0.82
            else:
                stagnation += 1
                # Failure: expand sigma slightly to re-explore
                sigma *= 1.10

            # Restart trigger: if stagnated, enlarge sigma and do a few uniform probes immediately
            if stagnation >= restart_patience and evals < budget:
                # Do 1/2 of remaining candidates (up to a cap)
                restart_batch = int(min(max(2, pop_size // 2), budget - evals))
                # Set sigma wide to jump
                sigma = min(sigma0, sigma * 2.0)
                C2 = lower + np.random.rand(restart_batch, dim) * (upper - lower)
                C2 = clip(C2)
                Y2 = np.empty(restart_batch, dtype=float)
                for i in range(restart_batch):
                    if evals >= budget:
                        Y2 = Y2[:i]
                        C2 = C2[:i]
                        break
                    Y2[i] = eval_point(C2[i])

                # Merge selection again with current pop
                if C2.shape[0] > 0:
                    U_n = X.shape[0] + C2.shape[0]
                    U_X = np.empty((U_n, dim), dtype=float)
                    U_Y = np.empty(U_n, dtype=float)
                    U_X[:X.shape[0]] = X
                    U_Y[:X.shape[0]] = Y
                    U_X[X.shape[0]:] = C2
                    U_Y[X.shape[0]:] = Y2
                    ord_u = np.argsort(U_Y)[:pop_size]
                    X = U_X[ord_u]
                    Y = U_Y[ord_u]

                # Reset stagnation after restart
                stagnation = 0

            # Prevent sigma from collapsing to zero or exploding too large
            # Use box scale as an upper bound.
            sigma = float(np.clip(sigma, 1e-12, sigma0 * 10.0))

        # Ensure best_x is defined
        if best_x is None:
            # Evaluate one feasible point if budget allowed (but budget likely >0).
            x = clip(center.copy())
            # If no evaluations were performed, evaluate once only if budget permits.
            if evals < budget:
                best_y = float(func(x))
                best_x = x
            else:
                best_x = x
                best_y = float("inf")

        return np.asarray(best_x, dtype=float), float(best_y)
