# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact, budget-aware derivative-free black-box
# minimizer for continuous spaces. It mixes global exploration using random
# directions with local exploitation via a simple coordinate/axis probing
# around the current best solution.
# Search state: Maintains the current best point x_best and its objective
# value y_best, along with a shrinking step size (radius) that adapts as
# improvements are found.
# Candidate generation: At each iteration, proposes a small set of candidate
# points by perturbing the current best along random directions and also along
# coordinate axes. Proposals are clipped to the feasible bounds.
# Selection and replacement: Evaluates each candidate, updates x_best/y_best
# if any candidate improves the objective (minimization).
# Adaptation: If improvement is observed, the step size increases slightly;
# otherwise it decreases. This simple success-based schedule helps balance
# exploration and exploitation.
# Exploration mechanisms: Random unit directions with stochastic scaling
# around the current best, plus a small portion of candidates sampled via
# uniform random points within the bounds.
# Exploitation mechanisms: Axis-aligned probing (both plus/minus) that refines
# around x_best in a coordinate-friendly way, especially helpful when the
# objective is sensitive along specific directions.
# Boundary handling: All candidate points are clipped to bounds. This guarantees
# feasibility without additional reflections or complex boundary models.
# Budget strategy: Strictly caps the number of objective evaluations. The algorithm
# converts the provided budget into a maximum number of calls and checks the
# remaining budget before every evaluation.
# Closest known influences: Combines ideas from evolution strategies (directional
# sampling), pattern search (axis probing), and success-based step-size control.
# Novelty or unusual aspects: Uses an adaptive radius with both random-direction
# and axis probes, while remaining fully budget-aware (no extra evaluations).
# Failure modes: For extremely noisy objectives or ill-conditioned landscapes,
# the simple step-size adaptation may converge prematurely. Also, if bounds are
# very tight, clipping can reduce effective exploration.
# ALGORITHM_ANALYSIS_NOTE_END

import numpy as np


class Algorithm:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = max(0, int(self.budget))
        if budget == 0:
            # No evaluations allowed; return a deterministic feasible point.
            lb, ub = self._get_bounds(func, dim)
            x = np.clip(np.zeros(dim), lb, ub)
            return x, float("inf")

        lb, ub = self._get_bounds(func, dim)
        span = ub - lb
        # Avoid division-by-zero issues; if span is 0, the variable is fixed.
        span_safe = np.where(span == 0, 1.0, span)

        evals = 0

        def eval_obj(x):
            nonlocal evals
            if evals >= budget:
                # Budget safety: should never happen due to checks, but keep robust.
                return float("inf")
            evals += 1
            y = func(x)
            return float(y)

        # Initial point: random uniform within bounds (harness sets seed).
        x_best = lb + np.random.rand(dim) * span_safe
        # If some dimensions have span 0, clip will set them correctly.
        x_best = np.clip(x_best, lb, ub)
        y_best = eval_obj(np.array(x_best, dtype=float))

        # Initial radius: a fraction of the bounds range.
        # If all spans are 0, radius becomes 0 and exploration is impossible.
        radius = 0.25 * float(np.mean(span_safe))
        radius = max(radius, 0.0)

        # Iteration budget allocation: we will perform batches of candidates.
        # Small batch size reduces the chance of exceeding budget.
        # Each candidate evaluation counts; we always check remaining budget.
        while evals < budget:
            remaining = budget - evals
            # Choose number of candidates for this iteration (including axis probes).
            # Keep it small to allow frequent adaptation.
            # Axis probing can contribute 2*dim candidates; we cap it.
            axis_cap = min(max(2, 2 * dim), 16)
            # Total candidates: random directions + optional random restart + axis probes
            n_rand_dirs = min(8, remaining)  # directional exploration
            n_random_restart = 1 if remaining >= 2 and np.random.rand() < 0.2 else 0
            n_axis = 0 if axis_cap == 0 else min(axis_cap, max(0, remaining - n_rand_dirs - n_random_restart))

            improved = False
            best_local_y = y_best
            best_local_x = x_best

            # --- Candidate generation: directional sampling around x_best ---
            # Use random unit directions; scale by radius and stochastic factors.
            for _ in range(n_rand_dirs):
                if evals >= budget:
                    break
                if radius <= 0.0:
                    break
                d = np.random.normal(size=dim)
                dn = np.linalg.norm(d)
                if dn == 0:
                    continue
                d = d / dn
                # Stochastic step: sometimes smaller, sometimes near radius.
                step = radius * (0.2 + 0.8 * np.random.rand())
                x = x_best + step * d
                x = np.clip(x, lb, ub)
                y = eval_obj(x)
                if y < best_local_y:
                    best_local_y, best_local_x = y, x
                    improved = True

            # --- Candidate generation: random restart (occasional global exploration) ---
            for _ in range(n_random_restart):
                if evals >= budget:
                    break
                # Sample uniformly within bounds.
                x = lb + np.random.rand(dim) * span_safe
                x = np.clip(x, lb, ub)
                y = eval_obj(x)
                if y < best_local_y:
                    best_local_y, best_local_x = y, x
                    improved = True

            # --- Candidate generation: axis-aligned probing around x_best ---
            # Probe a subset of coordinates to remain budget-aware.
            if n_axis > 0 and radius > 0.0 and dim > 0:
                # Select coordinates deterministically-ish based on a shuffle.
                coords = np.random.permutation(dim)[:n_axis // 2 + (n_axis % 2)]
                # Use plus/minus steps. If budget is tight, stop early.
                for idx in coords:
                    if evals >= budget:
                        break
                    step = radius * (0.2 + 0.8 * np.random.rand())
                    # Plus
                    x = np.array(best_local_x, copy=True)
                    x[idx] = np.clip(x[idx] + step, lb[idx], ub[idx])
                    y = eval_obj(x)
                    if y < best_local_y:
                        best_local_y, best_local_x = y, x
                        improved = True
                    if evals >= budget:
                        break
                    # Minus
                    x = np.array(best_local_x, copy=True)
                    x[idx] = np.clip(x[idx] - step, lb[idx], ub[idx])
                    y = eval_obj(x)
                    if y < best_local_y:
                        best_local_y, best_local_x = y, x
                        improved = True

            # --- Selection and replacement ---
            x_best, y_best = best_local_x, best_local_y

            # --- Adaptation: update radius based on success ---
            # Success: slightly expand; failure: shrink more.
            if improved:
                # Gentle expansion capped by bounds span.
                avg_span = float(np.mean(span_safe))
                radius = min(avg_span * 0.5, radius * 1.15 + 1e-12)
            else:
                radius *= 0.7

            # If radius becomes extremely small, stop exploring further.
            if radius <= 1e-14:
                # Still can evaluate random points if budget remains.
                # But avoid infinite loop; let while condition terminate naturally.
                if evals >= budget:
                    break

        return np.array(x_best, dtype=float), float(y_best)

    def _get_bounds(self, func, dim):
        # Try the known patterns: func.lower/func.upper or func.bounds.lb/ub.
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(getattr(func, "lower"), dtype=float)
            ub = np.asarray(getattr(func, "upper"), dtype=float)
        elif hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            if hasattr(b, "lb") and hasattr(b, "ub"):
                lb = np.asarray(b.lb, dtype=float)
                ub = np.asarray(b.ub, dtype=float)
            else:
                raise AttributeError("func.bounds must have lb and ub.")
        else:
            raise AttributeError("func must have lower/upper or bounds.lb/bounds.ub.")

        # Robust shape handling for dim=1 and vector dims.
        if lb.shape == () and dim == 1:
            lb = np.asarray([float(lb)], dtype=float)
            ub = np.asarray([float(ub)], dtype=float)

        lb = np.asarray(lb, dtype=float).reshape(-1)
        ub = np.asarray(ub, dtype=float).reshape(-1)

        if lb.size != dim or ub.size != dim:
            raise ValueError(f"Bounds dimension mismatch: expected {dim}, got {lb.size}/{ub.size}.")

        # Ensure lb <= ub.
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        return lb2, ub2
