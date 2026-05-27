# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer using a
# budgeted random local search with adaptive step sizes. It maintains a current
# best solution and a simple population of candidates to explore around it.
# Search state: Tracks current best (best_x, best_y), a step scale (sigma),
# and a small set of candidate points evaluated per iteration. Also tracks how
# many evaluations remain to never exceed the budget.
# Candidate generation: Each iteration samples candidates by adding Gaussian
# noise scaled by sigma to the current best, then also samples a few
# diversified points from random directions around the center to avoid getting
# stuck. All candidates are clipped to feasible bounds.
# Selection and replacement: Among newly evaluated candidates, the best (lowest)
# objective is selected. If it improves the current best, the step size is
# slightly reduced (more exploitation). If not, sigma is increased (more
# exploration).
# Adaptation: sigma adapts based on improvement using a simple multiplicative
# rule and is also bounded by problem scale to prevent numerical issues.
# Exploration mechanisms: Diversified random candidates plus an occasional
# "restart-like" perturbation when the search stagnates.
# Exploitation mechanisms: Gaussian perturbations centered at the best found so
# far with decreasing sigma upon improvements.
# Boundary handling: Candidate points are clipped to [lb, ub] from func or its
# bounds attribute.
# Budget strategy: The number of objective evaluations per iteration is capped
# so the total evaluations never exceeds the provided budget. The algorithm
# terminates immediately when the budget is exhausted.
# Closest known influences: Inspired by evolution strategies / CMA-like patterns
# but simplified to keep the module compact and standard-library only.
# Novelty or unusual aspects: Uses an adaptive sigma with a small batch strategy
# and deterministic budget accounting; includes diversified sampling and a
# lightweight stagnation trigger without external dependencies.
# Failure modes: If the objective is extremely noisy or highly non-smooth,
# the step adaptation may oscillate. If bounds are very tight, exploration may
# be limited and progress can stall; clipping can cause many points to land on
# boundaries.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            x0 = np.zeros(dim, dtype=float)
            return x0, float("inf")

        lb, ub = self._get_bounds(func, dim)
        rng = np.random  # harness seeds numpy before each run

        # Problem scale for reasonable step sizes.
        span = ub - lb
        # Avoid zero span: if a dimension is fixed, span becomes 1 for scaling.
        scale = np.where(span > 0, span, 1.0)

        # Choose initial center: best among a few random points within bounds.
        n_init = min(5, budget)
        n_init = max(1, n_init)

        evals = 0
        best_x = None
        best_y = float("inf")

        # Random points in the box: lb + u*(ub-lb)
        for _ in range(n_init):
            x = lb + rng.rand(dim) * span
            y = self._safe_eval(func, x)
            evals += 1
            if y < best_y or best_x is None:
                best_x, best_y = x, y
            if evals >= budget:
                return best_x, best_y

        # Adaptive step size around the best.
        # Start with a fraction of the span, but not too tiny.
        sigma = 0.25 * scale
        # Hard bounds on sigma to avoid numerical blow-up/underflow.
        sigma_min = 1e-12
        sigma_max = np.where(scale > 0, 2.0 * scale, 1.0)

        # How many candidates we try per iteration (batched but sequentially evaluated).
        # Keep it small to reduce overhead.
        batch = max(2, min(8, budget // 10 if budget >= 10 else budget))
        batch = min(batch, budget - evals) if budget - evals > 0 else 0

        # Stagnation counter triggers occasional diversified perturbation.
        stagnation = 0
        last_improve_evals = evals

        # Main loop
        while evals < budget:
            remaining = budget - evals
            if remaining <= 0:
                break

            # Adjust batch to remaining budget
            k = min(batch, remaining)
            # For diversity, add at least 1 random-direction candidate sometimes.
            # Split: exploitation candidates around best + diversification.
            n_exploit = max(1, k - 1)
            n_div = k - n_exploit

            candidates = []

            # Exploitation: Gaussian steps around best (coordinate-wise sigma).
            # Use a couple of different noise strengths to increase chance of progress.
            for _ in range(n_exploit):
                # Random normal perturbation
                step = rng.randn(dim) * sigma
                x = best_x + step
                x = self._clip(x, lb, ub)
                candidates.append(x)

            # Diversification: sample from a wider radius using random direction and mixing
            # with a random point in the domain.
            for _ in range(n_div):
                u = rng.rand(dim)
                x_rand = lb + u * span
                # Mix best with random point and also add a large perturbation
                mix = rng.rand() * 0.8  # mostly best, sometimes far
                step = rng.randn(dim) * (2.0 * sigma)
                x = (1.0 - mix) * best_x + mix * x_rand + step
                x = self._clip(x, lb, ub)
                candidates.append(x)

            # Evaluate candidates sequentially, respecting budget.
            improved = False
            for x in candidates:
                if evals >= budget:
                    break
                y = self._safe_eval(func, x)
                evals += 1
                if y < best_y:
                    best_x, best_y = x, y
                    improved = True

            # Adapt sigma based on improvement
            if improved:
                stagnation = 0
                last_improve_evals = evals
                # More exploitation: shrink sigma modestly
                sigma = np.maximum(sigma_min, 0.82 * sigma)
            else:
                stagnation += 1
                # More exploration: enlarge sigma modestly
                sigma = np.minimum(sigma_max, 1.15 * sigma)

            # Stagnation trigger: occasional "restart-like" perturbation of best.
            # Uses a budget-safe single evaluation.
            # (Doesn't overwrite best unless it improves.)
            if evals < budget and stagnation >= 6:
                x = best_x + rng.randn(dim) * (3.0 * sigma)
                x = self._clip(x, lb, ub)
                y = self._safe_eval(func, x)
                evals += 1
                if y < best_y:
                    best_x, best_y = x, y
                    stagnation = 0
                    sigma = np.maximum(sigma_min, 0.9 * sigma)
                else:
                    # If still no improvement, back off by enlarging slightly.
                    sigma = np.minimum(sigma_max, 1.2 * sigma)
                last_improve_evals = evals

        return best_x, best_y

    def _get_bounds(self, func: Any, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        # Bounds may be in func.lower/upper or func.bounds.lb/ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(getattr(func, "lower"), dtype=float)
            ub = np.asarray(getattr(func, "upper"), dtype=float)
        elif hasattr(func, "bounds") and hasattr(func.bounds, "lb") and hasattr(func.bounds, "ub"):
            lb = np.asarray(func.bounds.lb, dtype=float)
            ub = np.asarray(func.bounds.ub, dtype=float)
        else:
            # If bounds are not provided, assume wide bounds around 0.
            # This is a fallback; the benchmark harness should normally provide bounds.
            lb = -5.0 * np.ones(dim, dtype=float)
            ub = 5.0 * np.ones(dim, dtype=float)

        if lb.shape == () or ub.shape == ():
            lb = np.full(dim, float(lb), dtype=float)
            ub = np.full(dim, float(ub), dtype=float)

        lb = lb.reshape(-1)
        ub = ub.reshape(-1)

        if lb.size != dim or ub.size != dim:
            # Robustly handle mismatched shapes by broadcasting if possible.
            if lb.size == 1:
                lb = np.full(dim, float(lb[0]), dtype=float)
            if ub.size == 1:
                ub = np.full(dim, float(ub[0]), dtype=float)
            if lb.size != dim or ub.size != dim:
                # Final fallback: crop/pad deterministically.
                if lb.size < dim:
                    lb = np.pad(lb, (0, dim - lb.size), mode="edge")
                else:
                    lb = lb[:dim]
                if ub.size < dim:
                    ub = np.pad(ub, (0, dim - ub.size), mode="edge")
                else:
                    ub = ub[:dim]

        # Ensure lb <= ub elementwise
        lb2 = np.minimum(lb, ub)
        ub2 = np.maximum(lb, ub)
        return lb2, ub2

    def _clip(self, x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(x, lb), ub)

    def _safe_eval(self, func: Callable[[np.ndarray], float], x: np.ndarray) -> float:
        # The harness provides valid objective; we still try to keep robustness
        # for unexpected types by coercing to float.
        y = func(x)
        return float(y)
