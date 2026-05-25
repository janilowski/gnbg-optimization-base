# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm for
# continuous domains. It uses a global "multi-start" of candidate points combined
# with local refinement via coordinate-wise probing. The search maintains and
# updates a current best solution while repeatedly sampling/perturbing around it.
#
# Search state: The algorithm tracks the current best point x_best, its objective
# value y_best, the evaluation counter n_evals, and a dynamic step size sigma
# that shrinks as improvements occur.
#
# Candidate generation: Each iteration generates a small population of candidate
# points. A fraction of candidates is sampled uniformly from the domain
# (exploration), while the remainder are created by perturbing x_best using a
# Gaussian distribution scaled by sigma (exploitation). Additionally, when close
# to x_best, coordinate-wise finite probes are used to explore local structure.
#
# Selection and replacement: All generated candidates are evaluated (without
# exceeding the budget). The best evaluated point replaces x_best. If no improvement
# is found in a stage, sigma is reduced to focus the search; if improvement occurs,
# sigma is increased moderately to encourage further progress.
#
# Adaptation: sigma starts as a fraction of the domain diameter and is updated
# multiplicatively based on whether the iteration produced an improvement.
#
# Exploration mechanisms: Uniform sampling across the whole box is used early and
# occasionally thereafter to avoid premature convergence. Gaussian perturbations
# centered at x_best provide guided global exploration.
#
# Exploitation mechanisms: Coordinate-wise probing around x_best tries to find
# better directions along individual dimensions using step sizes proportional to
# sigma.
#
# Boundary handling: Every candidate is clipped to the feasible bounds after
# perturbation/probing, ensuring feasibility even for aggressive steps.
#
# Budget strategy: The algorithm never evaluates more than budget(func_calls) by
# tracking n_evals and truncating the number of candidates/probes accordingly.
# It makes progress in multiple "stages" where the number of iterations is chosen
# from the budget and dimension.
#
# Closest known influences: The design blends elements reminiscent of evolutionary
# strategies (sampling around a best point), random restarts / multi-start
# behavior, and simple coordinate search local probing.
#
# Novelty or unusual aspects: The combination of a bounded ES-like sampling loop
# with a lightweight coordinate probe step, all governed by a strict budget-aware
# evaluation counter.
#
# Failure modes: In very small budgets or highly ill-conditioned functions, the
# algorithm may spend too few evaluations on the most promising region. If the
# objective is extremely noisy, sigma updates based on single evaluations may
# become unstable; however, clipping and controlled step-size adaptation help.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Determine bounds from func interface.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError(
                "Function must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
            )

        if lb.shape != (dim,) or ub.shape != (dim,):
            # Attempt to broadcast/reshape if possible; otherwise raise.
            lb = np.reshape(lb, (dim,))
            ub = np.reshape(ub, (dim,))

        # Guard against pathological bounds; assume lb <= ub.
        width = ub - lb
        width = np.where(np.isfinite(width), width, 0.0)
        # For zero-width dimensions, perturbations should be zero.
        safe_width = np.where(width > 0, width, 0.0)

        def clip_to_bounds(x):
            return np.minimum(ub, np.maximum(lb, x))

        # Budget-aware evaluation.
        n_evals = 0

        def eval_x(x):
            nonlocal n_evals
            if n_evals >= budget:
                # Should never happen if logic is correct.
                return np.inf
            x = clip_to_bounds(np.asarray(x, dtype=float))
            y = func(x)
            n_evals += 1
            return float(y)

        # If dim is 0, trivially return empty vector.
        if dim == 0:
            return np.array([], dtype=float), float(eval_x(np.array([], dtype=float)))

        # Initialize: start with a few random samples, budget-aware.
        # Choose population size small enough to stay safe.
        # Multi-start improves robustness across dimensions.
        pop0 = 1 + min(6, budget)  # small initial exploration
        x_best = None
        y_best = np.inf

        # Initial step size: fraction of typical scale.
        # Use average non-zero width; fallback to 1.
        base_scale = float(np.mean(safe_width[safe_width > 0])) if np.any(safe_width > 0) else 1.0
        sigma = 0.25 * base_scale + 1e-12

        for _ in range(pop0):
            if n_evals >= budget:
                break
            r = np.random.rand(dim)
            x = lb + r * (ub - lb)
            y = eval_x(x)
            if y < y_best or x_best is None:
                x_best, y_best = x, y

        if x_best is None:
            # Should not happen unless budget==0, but keep safe.
            x_best = np.clip((lb + ub) * 0.5, lb, ub)
            y_best = eval_x(x_best)

        # Determine number of main iterations from budget.
        # Each iteration uses a small batch + occasional coordinate probes.
        # Keep iteration count modest to reduce overhead.
        remaining = budget - n_evals
        if remaining <= 0:
            return x_best, y_best

        # Batch size grows slightly with dim but stays small.
        # Ensure at least 1 candidate per iteration.
        batch = max(2, min(10, 2 + dim // 3))
        # Coordinate probes: probe a subset of dimensions each stage.
        # We'll probe up to k dims per stage depending on budget and dim.
        k_coords = max(1, min(dim, 1 + dim // 5))

        # Estimate iterations.
        # Each iteration evaluates up to batch candidates plus up to 2*k_coords probes.
        max_evals_per_iter = batch + 2 * k_coords
        iters = max(1, remaining // max_evals_per_iter)
        iters = min(iters, 200)  # safety cap

        for t in range(iters):
            if n_evals >= budget:
                break

            # Exploration/exploitation mixture.
            # Decrease uniform exploration over time.
            progress = t / max(1, iters - 1)
            p_uniform = max(0.05, 0.35 * (1.0 - progress))

            # Generate candidates.
            cand = []
            for j in range(batch):
                if n_evals >= budget:
                    break
                if np.random.rand() < p_uniform:
                    # Global exploration: uniform draw.
                    r = np.random.rand(dim)
                    x = lb + r * (ub - lb)
                else:
                    # Local exploitation: Gaussian perturbation around best.
                    # Scale per-dimension by width where possible.
                    scale_vec = np.where(safe_width > 0, safe_width, 1.0)
                    z = np.random.randn(dim)
                    x = x_best + (sigma * z) * (scale_vec / (base_scale + 1e-12))
                cand.append(clip_to_bounds(x))

            # Evaluate candidates in batch.
            improved = False
            best_local_x = x_best
            best_local_y = y_best

            for x in cand:
                if n_evals >= budget:
                    break
                y = eval_x(x)
                if y < best_local_y:
                    best_local_x, best_local_y = x, y
                    improved = True

            # Lightweight coordinate probing to refine locally.
            # Probe a subset (deterministic shuffle via RNG).
            if n_evals < budget:
                # Choose which dimensions to probe (random subset).
                if dim <= k_coords:
                    dims = np.arange(dim)
                else:
                    dims = np.random.choice(dim, size=k_coords, replace=False)

                # Probe both positive and negative directions along each chosen dim.
                # Step uses sigma and local scale on that coordinate.
                for d in dims:
                    if n_evals >= budget:
                        break
                    coord_scale = safe_width[d] if safe_width[d] > 0 else base_scale
                    step = sigma * (coord_scale / (base_scale + 1e-12))
                    if step == 0:
                        continue

                    x_plus = x_best.copy()
                    x_plus[d] = x_plus[d] + step
                    y_plus = eval_x(x_plus)
                    if y_plus < best_local_y:
                        best_local_x, best_local_y = x_plus, y_plus
                        improved = True

                    if n_evals >= budget:
                        break
                    x_minus = x_best.copy()
                    x_minus[d] = x_minus[d] - step
                    y_minus = eval_x(x_minus)
                    if y_minus < best_local_y:
                        best_local_x, best_local_y = x_minus, y_minus
                        improved = True

            # Update best and adapt sigma.
            if best_local_y < y_best:
                x_best, y_best = best_local_x, best_local_y

            # Adapt sigma: shrink when not improving; expand slightly when improving.
            if improved:
                # Mild expansion encourages exploration around a new basin.
                sigma = min(1.0 * base_scale + 1e-12, sigma * 1.15)
            else:
                # Shrink to focus exploitation and mitigate random wandering.
                sigma = max(1e-9, sigma * 0.85)

        return x_best, y_best
