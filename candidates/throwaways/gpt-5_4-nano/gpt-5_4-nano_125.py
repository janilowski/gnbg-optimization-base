# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# for an unknown objective function using a mixture of randomized sampling,
# coordinate-wise local refinement, and a simple restart strategy.
#
# Search state: The algorithm maintains a current best solution (best_x,
# best_y) and keeps a small population of candidate points sampled around the
# current best. It also tracks a step size (sigma) that shrinks when progress
# is made and grows/reset when stagnation is detected.
#
# Candidate generation: Each iteration samples candidates by taking the best
# point and adding Gaussian perturbations scaled by sigma. Additionally,
# it can generate coordinate-wise probes (one dimension at a time) with a
# larger step to help escape shallow local minima.
#
# Selection and replacement: For each batch of sampled points, the algorithm
# evaluates each candidate and selects the best one; if it improves upon
# best_y, it becomes the new incumbent. The population is effectively
# replaced by resampling around the incumbent each iteration.
#
# Adaptation: Sigma is reduced when an improvement is observed (successful
# exploitation) and increased or reset when no improvement is seen (promotes
# exploration / restarts).
#
# Exploration mechanisms: Early iterations use larger sigma and random
# perturbations; coordinate-wise probes add directional exploration without
# requiring gradients.
#
# Exploitation mechanisms: When improvements occur, sigma shrinks and
# candidates concentrate around the current best point.
#
# Boundary handling: All candidates are clipped to provided box constraints.
# Bounds are read from func.lower/func.upper or func.bounds.lb/func.bounds.ub.
#
# Budget strategy: The algorithm strictly limits the number of function
# evaluations to the provided budget. It estimates evaluations per iteration
# and stops once the budget is reached.
#
# Closest known influences: The design resembles a lightweight evolution
# strategy / CMA-free black-box optimizer with local refinement, but uses only
# simple isotropic sampling plus coordinate probes.
#
# Novelty or unusual aspects: Uses an evaluation-budget-aware loop that
# chooses batch sizes adaptively, and mixes isotropic Gaussian sampling with
# one-off coordinate probes for robustness across dimensions.
#
# Failure modes: If the objective is extremely noisy or highly deceptive, the
# algorithm may stagnate; the restart mechanism and adaptive sigma mitigate
# this but cannot guarantee global optimality.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _get_bounds(self, func):
        # Prefer func.lower/func.upper; otherwise use func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            raise AttributeError(
                "Objective must provide bounds via func.lower/func.upper "
                "or func.bounds.lb/func.bounds.ub."
            )

        lb = np.broadcast_to(lb, (self.dim,)).copy()
        ub = np.broadcast_to(ub, (self.dim,)).copy()

        # Ensure lb <= ub, swap if needed.
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        return lo, hi

    def __call__(self, func):
        lb, ub = self._get_bounds(func)
        d = self.dim

        # Defensive checks.
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if d <= 0:
            raise ValueError("dim must be positive")

        rng = np.random  # harness sets numpy seed globally

        evals = 0

        def clip(x):
            return np.minimum(ub, np.maximum(lb, x))

        def eval_one(x):
            nonlocal evals
            # Enforce budget strictly.
            if evals >= self.budget:
                raise RuntimeError("Evaluation budget exceeded (internal).")
            y = func(np.asarray(x, dtype=float))
            evals += 1
            return float(np.asarray(y))

        # Initialize: sample a few random points to find a reasonable incumbent.
        # Number of initial samples is budget-aware.
        # Use small constant for robustness; ensure at least 1.
        n_init = min(max(1, d + 1), self.budget)
        best_x = None
        best_y = None

        for _ in range(n_init):
            x = lb + (ub - lb) * rng.random(d)
            y = eval_one(x)
            if best_y is None or y < best_y:
                best_x, best_y = x, y

        # Base step size: fraction of average range.
        span = ub - lb
        # If span is degenerate, keep sigma small but nonzero.
        avg_span = float(np.mean(span)) if np.all(np.isfinite(span)) else float(np.mean(np.abs(span)))
        if avg_span <= 0:
            # If all bounds equal, only one feasible point exists.
            return clip(best_x), best_y

        sigma = 0.3 * avg_span
        sigma_min = 1e-12 * max(1.0, avg_span)
        sigma_max = 1.0 * avg_span

        # Stagnation control.
        stagnation = 0
        best_overall_y = best_y

        # Iterative search with evaluation-budget-aware batches.
        # Each loop: sample k candidates; optional coordinate probes; select best.
        while evals < self.budget:
            remaining = self.budget - evals
            if remaining <= 0:
                break

            # Choose batch size: larger near beginning, smaller near end.
            # Keep it modest to reduce overhead.
            k = min(max(4, d + 2), remaining)
            # If close to budget, reduce batch size to leave room for probes.
            k = max(1, min(k, remaining - 1)) if remaining > 2 else remaining

            # Exploration+exploitation: isotropic Gaussian around best_x.
            # Use a mix of normal perturbations at slightly different scales.
            # (Two scales helps handle varying landscape steepness.)
            scale_a = sigma
            scale_b = sigma * 0.25 if sigma > sigma_min else sigma

            candidates = np.empty((k, d), dtype=float)
            # Create perturbations efficiently.
            # Half candidates with larger noise, half with smaller.
            n_a = (k + 1) // 2
            n_b = k - n_a
            if n_a > 0:
                candidates[:n_a] = best_x + scale_a * rng.randn(n_a, d)
            if n_b > 0:
                candidates[n_a:] = best_x + scale_b * rng.randn(n_b, d)

            # Clip to bounds.
            for i in range(k):
                candidates[i] = clip(candidates[i])

            # Evaluate batch and find best.
            batch_best_x = None
            batch_best_y = None
            for i in range(k):
                if evals >= self.budget:
                    break
                x = candidates[i]
                y = eval_one(x)
                if batch_best_y is None or y < batch_best_y:
                    batch_best_x, batch_best_y = x, y

            # Optional coordinate probes for robustness.
            # Use a couple of dimensions per iteration (budget-aware).
            # Probes are noiseless directional checks.
            improved = False
            if batch_best_y is not None and batch_best_y < best_y:
                best_x, best_y = batch_best_x, batch_best_y
                improved = True

            # Decide whether to do coordinate probing.
            remaining = self.budget - evals
            if remaining > 0 and (not improved) and (d > 1):
                # Probe at most min(2*d, remaining) evaluations but cap to be small.
                # We'll choose a few coordinates based on random subset.
                # Coordinate step larger than current sigma slightly.
                coord_step = max(sigma * 1.5, 1e-8 * avg_span)
                # Determine how many probes pairs we can afford: each pair uses 2 evals.
                max_pairs = remaining // 2
                if max_pairs > 0:
                    # Use at least 1 pair if possible, else skip.
                    n_pairs = min(max_pairs, max(1, min(3, d // 2 + 1)))
                    # Choose random coordinates (or include some deterministic ones).
                    idx = rng.choice(d, size=n_pairs, replace=False) if d >= n_pairs else np.arange(d)
                    for j in idx:
                        if evals + 2 > self.budget:
                            break
                        e = np.zeros(d, dtype=float)
                        e[j] = coord_step
                        x1 = clip(best_x + e)
                        x2 = clip(best_x - e)
                        y1 = eval_one(x1)
                        if y1 < best_y:
                            best_x, best_y = x1, y1
                            improved = True
                        y2 = eval_one(x2)
                        if y2 < best_y:
                            best_x, best_y = x2, y2
                            improved = True

            # Adapt sigma and stagnation.
            if improved:
                # Successful exploitation: shrink sigma.
                stagnation = 0
                best_overall_y = min(best_overall_y, best_y)
                sigma = max(sigma_min, sigma * 0.85)
            else:
                stagnation += 1
                # Mild increase to escape local minima.
                sigma = min(sigma_max, sigma * 1.08)

                # Restart mechanism if prolonged stagnation.
                # Restart: reset around a new random point and reset sigma.
                if stagnation >= max(5, d // 2):
                    # Evaluate one or two random points to seed a better region,
                    # without exceeding the budget.
                    remaining = self.budget - evals
                    if remaining <= 0:
                        break
                    n_restart = 1 if remaining < 2 else 2
                    seed_best_x = best_x
                    seed_best_y = best_y
                    for _ in range(n_restart):
                        if evals >= self.budget:
                            break
                        x = lb + (ub - lb) * rng.random(d)
                        y = eval_one(x)
                        if y < seed_best_y:
                            seed_best_x, seed_best_y = x, y
                    best_x, best_y = seed_best_x, seed_best_y
                    sigma = 0.3 * avg_span
                    stagnation = 0

            # Update best record.
            if best_y < best_overall_y:
                best_overall_y = best_y

            # If sigma is already tiny, still continue until budget ends
            # (may help with flat landscapes); but avoid wasted work by ensuring
            # sigma can't become exactly zero due to clipping.
            if sigma < sigma_min:
                sigma = sigma_min

        return clip(best_x), float(best_y)
