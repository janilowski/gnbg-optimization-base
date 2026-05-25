# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# that works on continuous bounded domains. It is designed to be robust in
# arbitrary dimensions while respecting a strict evaluation budget.
# Search state: The algorithm maintains a current incumbent solution (best_x,
# best_y), a candidate center (often the incumbent), and a step-size vector
# sigma that controls the scale of perturbations.
# Candidate generation: Each iteration samples a small batch of candidates by
# adding Gaussian noise with per-dimension step sizes to the current center.
# It also includes a simple coordinate-wise probing strategy to refine local
# directionality when improvement is found.
# Selection and replacement: Candidates are evaluated, and the best among them
# is selected. If a candidate improves the incumbent, it replaces the
# incumbent. Otherwise, the algorithm relies on shrinking/expanding sigma to
# control exploration versus exploitation.
# Adaptation: The step sizes (sigma) adapt based on whether improvements are
# observed: sigma shrinks after improvement (more exploitation) and expands
# after stagnation (more exploration).
# Exploration mechanisms: Stochastic sampling around the incumbent and periodic
# larger-radius probes help the algorithm escape local minima.
# Exploitation mechanisms: On improvement, the algorithm moves the search
# center to the new best and reduces sigma to focus the search locally.
# Boundary handling: Candidates are clipped to the feasible bounds; sigma is
# also bounded away from zero to avoid numerical issues.
# Budget strategy: The algorithm keeps an internal evaluation counter and never
# calls the objective more than `budget` times. It uses a final "best-so-far"
# return without extra evaluations.
# Closest known influences: The approach is inspired by simplified evolution
# strategies / CMA-like step-size adaptation, combined with basic local probing.
# Novelty or unusual aspects: It uses a per-dimension sigma with a simple,
# budget-aware batch evaluation and occasional coordinate probing to reduce
# wasted evaluations in high dimensions.
# Failure modes: If the objective is extremely noisy or the bounds are very
# tight, sigma adaptation may oscillate or converge prematurely; in such cases,
# exploration probes and sigma expansion mitigate the risk but cannot guarantee
# optimality.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        n = self.dim
        max_evals = self.budget
        if max_evals <= 0:
            # No evaluations allowed: return a deterministic feasible point.
            lb, ub = self._get_bounds(func)
            x0 = np.clip((lb + ub) / 2.0, lb, ub)
            return x0, float("inf")

        lb, ub = self._get_bounds(func)
        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if lb.shape[0] != n or ub.shape[0] != n:
            raise ValueError("Bounds dimensionality does not match dim.")

        # Feasible initial point: mid-point of bounds.
        center = np.clip((lb + ub) / 2.0, lb, ub)

        # Initial step sizes: a fraction of the domain width.
        width = np.maximum(ub - lb, 1e-12)
        sigma = 0.25 * width
        sigma = np.clip(sigma, 1e-12, width)

        evals = 0
        best_x = None
        best_y = None

        def f(x):
            nonlocal evals, best_x, best_y
            # Ensure never exceed budget.
            if evals >= max_evals:
                # Should not happen; return an arbitrary large value.
                return float("inf")
            y = float(func(x))
            evals += 1
            if best_y is None or y < best_y:
                best_y = y
                best_x = x.copy()
            return y

        # Evaluate initial center.
        y_center = f(center)

        # Budget-aware iteration planning.
        # We'll use small batches to improve efficiency while maintaining simplicity.
        # Remaining evaluations govern the number of iterations.
        remaining = max_evals - evals
        if remaining <= 0:
            return best_x, best_y

        # Batch size: scales lightly with dimension but bounded.
        # Keep it modest to reduce overhead and allow more iterations.
        batch = int(np.clip(4 + n // 10, 4, 16))
        batch = min(batch, remaining)

        # Random generator (harness sets numpy seed).
        rng = np.random

        # Exploration/exploitation schedule.
        # sigma_min prevents collapse; sigma_max prevents out-of-bounds dominance.
        sigma_min = 1e-12 * width
        sigma_max = 0.8 * width

        # Coordinate probing frequency.
        probe_every = max(1, n // 5)

        iter_id = 0
        while evals < max_evals:
            iter_id += 1
            remaining = max_evals - evals
            if remaining <= 0:
                break

            b = min(batch, remaining)

            # --- Candidate generation (Gaussian mutations around center) ---
            # Use per-dimension sigma scaling.
            # Candidate x = center + sigma * N(0,1)
            noise = rng.standard_normal(size=(b, n))
            candidates = center[None, :] + noise * sigma[None, :]

            # --- Boundary handling: clip to feasible box ---
            candidates = np.clip(candidates, lb, ub)

            # Evaluate candidates and find best in this batch.
            local_best_x = None
            local_best_y = None
            for i in range(b):
                x_i = candidates[i]
                y_i = f(x_i)
                if local_best_y is None or y_i < local_best_y:
                    local_best_y = y_i
                    local_best_x = x_i

            improved = (local_best_y is not None) and (local_best_y < best_y - 1e-15)

            # --- Adaptation ---
            if improved:
                # Exploit: move center to local best and shrink sigma.
                center = local_best_x.copy()
                # Aggressive shrink on improvements.
                sigma = np.maximum(sigma * 0.8, sigma_min)
            else:
                # Explore: keep center at incumbent best and expand sigma slightly.
                if best_x is not None:
                    center = best_x.copy()
                sigma = np.minimum(sigma * 1.07, sigma_max)

            # --- Occasional coordinate probing ---
            # If stagnating, try structured moves along a few coordinates.
            if (iter_id % probe_every == 0) and (evals < max_evals):
                # Choose up to K coordinates, bias towards largest sigma dimensions.
                K = int(np.clip(3 + n // 20, 3, 8))
                k = min(K, n)
                # Pick indices deterministically based on current sigma magnitudes.
                idx = np.argsort(-sigma)[:k]

                # Evaluate +/- steps for selected coordinates, budget-aware.
                # Each coordinate may use up to 2 evaluations.
                # We stop when budget is close.
                for j in idx:
                    if evals >= max_evals:
                        break
                    step = sigma[j]
                    # Two directions
                    for sgn in (1.0, -1.0):
                        if evals >= max_evals:
                            break
                        x_probe = center.copy()
                        x_probe[j] = np.clip(x_probe[j] + sgn * step, lb[j], ub[j])
                        y_probe = f(x_probe)
                        if best_y is not None and y_probe < best_y - 1e-15:
                            center = best_x.copy()
                            # When probing helps, shrink more conservatively.
                            sigma = np.maximum(sigma * 0.85, sigma_min)

                # If no improvement came from probes, slightly expand sigma.
                # (This is safe because we checked improvement implicitly via best_x.)
                if center is not None and best_x is not None:
                    # Compare if center == best_x is typical even without improvement.
                    # Use a heuristic: if sigma is too small, expand a bit; else keep.
                    if np.all(sigma <= 1.01 * sigma_min):
                        sigma = np.minimum(sigma * 1.15, sigma_max)

        # Return the best found within budget.
        return best_x, best_y

    @staticmethod
    def _get_bounds(func):
        # Preferred interfaces:
        # - func.lower / func.upper (array-like)
        # - func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return np.asarray(func.lower, dtype=float), np.asarray(func.upper, dtype=float)
        if hasattr(func, "bounds"):
            b = func.bounds
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return np.asarray(b.lb, dtype=float), np.asarray(b.ub, dtype=float)
        raise AttributeError(
            "Objective must provide bounds via func.lower/func.upper or func.bounds.lb/func.bounds.ub."
        )
