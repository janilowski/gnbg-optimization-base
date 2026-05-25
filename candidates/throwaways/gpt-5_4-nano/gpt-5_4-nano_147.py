# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm
# (derivative-free, single-point evaluations) using a population-based
# evolutionary strategy with self-adaptive mutation scale and a local
# coordinate pattern search around the current best. It is robust across
# dimensions by using vectorized Gaussian sampling and a budget-aware loop.
#
# Search state: Tracks current evaluation budget, best-so-far point/value,
# a mutation step-size (sigma) that adapts, and a small population of
# candidate solutions each iteration.
#
# Candidate generation: In each iteration, it samples a population of
# candidates by adding Gaussian noise scaled by sigma to the current
# best (and optionally the current best/mean). It also occasionally adds
# structured "axis steps" (coordinate pattern moves) to encourage progress
# even when gradients are absent.
#
# Selection and replacement: For each batch, candidates are evaluated and the
# best candidate replaces the incumbent best. The population is not persisted
# across iterations; instead, sigma adapts based on whether improvements occur.
#
# Adaptation: Sigma increases slightly after an unsuccessful iteration
# (to encourage exploration) and decreases when improvements are found
# (to focus exploitation). The adaptation is bounded to keep steps stable.
#
# Exploration mechanisms: Random Gaussian sampling with an adaptive sigma,
# plus occasional coordinate pattern probes in random coordinate directions.
#
# Exploitation mechanisms: Sampling around the best and coordinate pattern
# search using a shrinking step, both biased toward the current best.
#
# Boundary handling: Inputs are clipped to the provided lower/upper bounds
# (from func.lower/upper or func.bounds.lb/ub). This ensures feasibility
# without wasting evaluations outside bounds.
#
# Budget strategy: The total number of objective evaluations is capped by the
# provided budget. The algorithm estimates remaining evaluations each loop and
# chooses a batch size that will not exceed the budget.
#
# Closest known influences: Combines ideas from (1) evolutionary strategies with
# step-size adaptation and (2) coordinate/pattern search, tailored for strict
# evaluation budgets.
#
# Novelty or unusual aspects: Uses a hybrid of population sampling and a
# lightweight coordinate pattern move that triggers based on recent progress,
# enabling fast local refinement when improvements are being found.
#
# Failure modes: If the objective is extremely noisy or highly non-smooth,
# adaptation may oscillate. With very tight budgets, performance may be close
# to random search. For badly scaled bounds, sigma initialization may be
# suboptimal (still handled via clipping and adaptive scaling).
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
            # No evaluations allowed; return a feasible arbitrary point.
            lb, ub = self._get_bounds(func, dim)
            x0 = (lb + ub) / 2.0
            return x0, float("inf")

        lb, ub = self._get_bounds(func, dim)
        rng = np.random

        # Initialize with mid-point and a few random probes if budget allows.
        x_best = np.clip((lb + ub) / 2.0, lb, ub)
        y_best = None

        eval_count = 0

        def f_eval(x):
            nonlocal eval_count, y_best
            # Objective is minimization; evaluate exactly once per call.
            y = float(func(x))
            eval_count += 1
            if y_best is None or y < y_best:
                y_best = y
                x_best = np.array(x, copy=True)
            return y

        # Set initial mutation scale based on bounds.
        span = np.maximum(ub - lb, 1e-12)
        # sigma0: 0.25 of typical span, but not too tiny/huge.
        sigma = 0.25 * float(np.median(span))
        sigma = float(np.clip(sigma, 1e-12, 1e6))

        # How many candidates per iteration (must be >=1).
        # Keep it small to remain budget-aware and robust across dims.
        base_pop = 2 + min(10, dim)
        coord_pop = 1 + min(3, dim)

        # Evaluate incumbent mid-point first.
        y_best = float(func(x_best))
        eval_count = 1

        # Progress tracking for adaptation and coordinate moves.
        improved_recently = True
        no_improve_steps = 0

        # Main loop: each loop consumes a batch; never exceed budget.
        while eval_count < budget:
            remaining = budget - eval_count

            # Choose batch size not exceeding remaining.
            pop_size = min(base_pop, remaining)

            # Construct a candidate set:
            # - Mostly around current best with Gaussian perturbations.
            # - A few additional points from occasional jitter around a random point.
            # Vectorized generation for speed.
            # Shape: (pop_size, dim)
            noise = rng.normal(size=(pop_size, dim))
            # Scale per-dimension using span for robustness.
            per_dim_scale = (span / float(np.sqrt(dim)))  # typical normalized scaling
            candidates = x_best + (sigma * noise) * (per_dim_scale / float(np.median(per_dim_scale)))
            candidates = np.clip(candidates, lb, ub)

            # Evaluate candidates sequentially to keep strict accounting.
            # Also allow early break if budget runs out (shouldn't due to pop_size clamp).
            best_y_iter = None
            best_x_iter = None
            for i in range(pop_size):
                y = float(func(candidates[i]))
                eval_count += 1
                if best_y_iter is None or y < best_y_iter:
                    best_y_iter = y
                    best_x_iter = candidates[i].copy()
                if eval_count >= budget:
                    break

            if best_y_iter is not None and best_y_iter < y_best:
                # Improvement found.
                x_best = best_x_iter
                y_best = best_y_iter
                improved_recently = True
                no_improve_steps = 0
                # Shrink sigma to exploit near the best.
                sigma *= 0.85
            else:
                improved_recently = False
                no_improve_steps += 1
                # Expand sigma slightly to explore.
                sigma *= 1.05

            # Clamp sigma to sensible range derived from bounds.
            # Lower bound ties to numerical and span.
            sigma = float(np.clip(sigma, 1e-12, 1e3 * float(np.max(span))))

            # Coordinate pattern exploration: do a few axis-aligned probes when stuck.
            # This helps in nonsmooth landscapes where pure ES may meander.
            if (not improved_recently) and no_improve_steps >= 2 and eval_count < budget:
                remaining = budget - eval_count
                num_axes = min(coord_pop, dim)
                num_axes = min(num_axes, remaining)  # evaluations for axes
                # Pick axes randomly.
                axes = rng.choice(dim, size=num_axes, replace=False)
                # Pattern step proportional to sigma and span.
                step = sigma * 0.5

                # Try +step then -step for each axis, but budget-aware.
                # Evaluate in a prioritized order: the direction that moves
                # candidates away/toward bounds more safely is handled by clipping.
                dirs = rng.choice([-1.0, 1.0], size=num_axes)
                for k in range(num_axes):
                    if eval_count >= budget:
                        break
                    d = dirs[k]
                    axis = int(axes[k])
                    x_try = np.array(x_best, copy=True)
                    x_try[axis] = x_try[axis] + d * step
                    x_try = np.clip(x_try, lb, ub)
                    y_try = float(func(x_try))
                    eval_count += 1
                    if y_try < y_best:
                        x_best = x_try
                        y_best = y_try
                        improved_recently = True
                        no_improve_steps = 0
                        # If we improved, reduce sigma to refine.
                        sigma *= 0.8

                # If still no improvement, slightly reduce step to avoid bouncing.
                if not improved_recently:
                    sigma *= 0.95

        return np.array(x_best, copy=True), float(y_best)

    @staticmethod
    def _get_bounds(func, dim):
        # Bounds can be exposed in several ways. The harness likely supplies one.
        # Priority:
        # 1) func.lower/func.upper
        # 2) func.bounds.lb / func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(getattr(func, "lower"), dtype=float)
            ub = np.asarray(getattr(func, "upper"), dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(getattr(func.bounds, "lb"), dtype=float)
            ub = np.asarray(getattr(func.bounds, "ub"), dtype=float)
        else:
            # Fallback: use a standard box if not provided.
            lb = -5.0 * np.ones(dim, dtype=float)
            ub = 5.0 * np.ones(dim, dtype=float)

        # Ensure correct shape/length.
        lb = lb.reshape(-1)
        ub = ub.reshape(-1)
        if lb.size != dim or ub.size != dim:
            # If bounds are scalars, broadcast.
            if lb.size == 1:
                lb = np.full(dim, float(lb[0]), dtype=float)
            if ub.size == 1:
                ub = np.full(dim, float(ub[0]), dtype=float)
            # Otherwise, best-effort trunc/pad.
            if lb.size != dim:
                lb = (np.resize(lb, dim)).astype(float)
            if ub.size != dim:
                ub = (np.resize(ub, dim)).astype(float)

        # Ensure lb <= ub.
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        return lb2, ub2
