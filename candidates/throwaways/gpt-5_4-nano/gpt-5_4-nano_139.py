# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact derivative-free black-box minimizer using a simple
# evolutionary strategy with self-adaptive step sizes plus a periodic coordinate
# local refinement. The algorithm is designed to be budget-aware and robust across
# dimensions.
# Search state: Maintains a population of candidate vectors (x), their objective
# values (y), and per-dimension step sizes (sigma) for mutation. Also tracks
# the current best solution found so far.
# Candidate generation: Each generation samples offspring by adding Gaussian noise
# scaled by sigma (and optionally a global step scale). Occasional "coordinate
# moves" are attempted around the best point to refine along axes.
# Selection and replacement: Uses (μ+λ) strategy: offspring are evaluated, then the
# next population is formed by selecting the best μ individuals among parents
# and offspring. The best-so-far is preserved.
# Adaptation: Step sizes are adapted based on success rate of offspring improving
# over the current best (rule-of-thumb). Sigma is shrunk when improvements are rare
# and expanded when improvements are frequent.
# Exploration mechanisms: Mutation with Gaussian noise plus periodic coordinate
# probing provide both global and local exploration.
# Exploitation mechanisms: Best-first survival (μ+λ) and coordinate refinement
# drive exploitation toward promising regions.
# Boundary handling: Uses hard clipping to bounds after mutation/probing.
# Budget strategy: Ensures the total number of objective evaluations never exceeds
# the provided budget by computing the maximum feasible number of generations and
# truncating offspring counts if needed.
# Closest known influences: Mixes ideas from CMA-ES-like (population + adaptation)
# and Evolution Strategies (1/5 success-ish adaptation), but implemented in a
# minimal, self-contained way without covariance learning.
# Novelty or unusual aspects: Adds a lightweight coordinate refinement stage that
# uses the current sigma to probe around the best point, improving convergence
# on separable or axis-aligned landscapes.
# Failure modes: If the objective is extremely noisy or non-smooth, adaptation
# may oscillate; clipping to bounds may cause premature stagnation at boundaries.
# Also, for very small budgets the method essentially behaves like a few random
# samples around the initial seed.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func: Any) -> Tuple[np.ndarray, float]:
        # --- Bounds reading ---
        lb, ub = self._get_bounds(func)
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)

        d = self.dim
        if lb.shape == ():
            lb = np.full(d, float(lb))
        if ub.shape == ():
            ub = np.full(d, float(ub))
        lb = lb.reshape(d)
        ub = ub.reshape(d)
        if np.any(ub <= lb):
            raise ValueError("Invalid bounds: require ub > lb elementwise.")

        # --- Budget bookkeeping ---
        max_evals = max(1, self.budget)
        evals = 0

        def clamp(x: np.ndarray) -> np.ndarray:
            return np.minimum(ub, np.maximum(lb, x))

        def eval_obj(x: np.ndarray) -> float:
            nonlocal evals
            if evals >= max_evals:
                # Should never happen if budget logic is correct; guard anyway.
                return float("inf")
            y = func(x)
            evals += 1
            # Ensure scalar float
            return float(np.asarray(y).item())

        # --- Initialization ---
        rng = np.random.default_rng()
        # Start near the middle plus some random jitter
        center = (lb + ub) / 2.0
        span = (ub - lb)
        # Typical initial sigma: 25% of range, but scaled down for small ranges
        base_sigma = 0.25 * span
        base_sigma = np.where(base_sigma > 0, base_sigma, 1.0)

        # Choose population sizes to fit budget.
        # We will evaluate: init_n + generations * offspring_n + (optional) coordinate tries.
        # Use a conservative offspring count to keep number of generations reasonable.
        init_n = min(10, max_evals)
        # μ and λ for (μ+λ): population maintained at μ
        mu = max(2, min(12, (init_n // 2) * 2))  # even-ish
        mu = min(mu, init_n)

        # If budget is tiny, just do random sampling.
        if max_evals <= 1:
            x0 = clamp(center + 0.0 * rng.normal(size=d))
            y0 = eval_obj(x0)
            return x0, y0

        # Ensure at least one init evaluation.
        mu = max(2, min(mu, max_evals))
        init_n = mu

        # Build initial population
        X = np.empty((mu, d), dtype=float)
        if mu == 1:
            X[0] = clamp(center)
        else:
            # include center and random points
            X[0] = clamp(center)
            if mu > 1:
                for i in range(1, mu):
                    r = rng.random(d)
                    X[i] = clamp(lb + r * (ub - lb))

        Y = np.empty(mu, dtype=float)
        best_idx = 0
        best_x = X[0].copy()
        best_y = eval_obj(X[0])
        best_idx = 0

        for i in range(1, mu):
            Y[i] = eval_obj(X[i])
            if Y[i] < best_y:
                best_y = Y[i]
                best_x = X[i].copy()

        # Per-dimension step sizes; keep within reasonable bounds
        sigma = base_sigma.copy()
        # If span is very small, sigma might be tiny; avoid zero
        sigma = np.where(sigma > 0, sigma, 1e-3)

        # --- Evolution loop ---
        # Determine number of generations we can afford.
        # Each generation evaluates λ offspring.
        # We'll set λ around 4..8 depending on budget, capped so we do some generations.
        # Also reserve some budget for coordinate refinement.
        # Coordinate refinement happens every k generations or if sigma is large enough.
        coord_reserve = min(0, max_evals - evals)  # default 0
        # reserve a small fraction for refinement attempts
        coord_reserve = int(max(0, 0.08 * (max_evals - evals)))
        coord_reserve = min(coord_reserve, max(0, max_evals - evals))

        # Remaining budget for offspring evaluations
        remaining = max(0, max_evals - evals - coord_reserve)
        if remaining <= 0:
            # Budget tight: do no further search
            return best_x, best_y

        # Pick λ so that we can have at least a few generations if budget allows
        # but not too many evaluations.
        lam = min(12, max(4, 2 * d // max(1, int(math.sqrt(max(1, d))))))
        lam = min(lam, remaining)  # can't exceed remaining
        lam = max(2, lam)

        # Number of generations
        gens = remaining // lam
        if gens <= 0:
            gens = 1
            lam = remaining if remaining > 0 else 1

        # Parent selection size: μ (already maintained)
        # Coordinate refinement settings
        coord_every = max(2, min(6, gens // 3 if gens >= 3 else 2))
        coord_tries = 0
        coord_budget = max(0, max_evals - evals)
        # allocate small budget for coordinate probing across whole run
        coord_budget = min(coord_budget, coord_reserve)
        # how many coordinate evaluations total: each coordinate try evaluates 2 candidates
        # We'll choose per-probe cap.
        per_coord_probe = max(1, min(d, 4 + d // 5))  # number of coordinates considered per probe

        # Helper to keep population sorted by fitness
        def sort_population(Xp: np.ndarray, Yp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            idx = np.argsort(Yp)
            return Xp[idx], Yp[idx]

        # Track success to adapt sigma
        success_window = 0
        window_size = max(1, gens // 3)
        success_threshold = 0.25  # heuristic

        # Global step scale: start at 1, adapt with sigma
        global_scale = 1.0

        for gen in range(gens):
            # --- Generate offspring ---
            # Offspring count fixed at lam for simplicity; budget is handled by global gens.
            offspring = np.empty((lam, d), dtype=float)
            offspring_y = np.empty(lam, dtype=float)

            # Normal mutation: x' = x_parent + N(0, sigma^2) * global_scale
            # Choose parents by tournament among current population
            # Tournament size small for speed
            tsize = 3 if mu >= 3 else 2

            for i in range(lam):
                # tournament select parent
                if mu == 1:
                    p = 0
                else:
                    cand = rng.integers(0, mu, size=tsize)
                    p = cand[np.argmin(Y[cand])]
                # Create mutant
                noise = rng.normal(0.0, 1.0, size=d)
                step = sigma * noise * global_scale
                x_new = clamp(X[p] + step)
                offspring[i] = x_new
                offspring_y[i] = eval_obj(x_new)

            # --- Selection (μ+λ): choose best μ among parents and offspring ---
            X_all = np.vstack((X, offspring))
            Y_all = np.concatenate((Y, offspring_y))
            X_all, Y_all = sort_population(X_all, Y_all)
            X = X_all[:mu]
            Y = Y_all[:mu]

            # Update best-so-far
            if Y[0] < best_y:
                best_y = float(Y[0])
                best_x = X[0].copy()

            # --- Adaptation based on improvements in this generation ---
            # Count offspring that beat current best_y_old (strictly).
            # Use pre-generation best_y_old; approximate with current best_y before update isn't kept.
            # We can compare offspring_y against best_y before this generation by reconstructing:
            # Instead, use comparisons against min(Y_all) from previous generation not available.
            # Simpler: success if any offspring improves the previous best by at least tiny margin.
            # We'll approximate by checking if offspring_y has values < min(prev_best_y, current_best_y + eps).
            # Use stored best_y_old at start of gen.
            # To avoid extra variables, we can recompute: compare offspring_y to current best_y + 1e-15
            # counts those that are <= current best may include parent. We'll instead track by
            # whether Y_all[0] came from offspring.
            # Determine whether the best element among offspring improved:
            best_off_idx = int(np.argmin(offspring_y)) if lam > 0 else 0
            improved = float(offspring_y[best_off_idx]) < best_y + 1e-15  # heuristic
            if improved and float(offspring_y[best_off_idx]) <= best_y + 1e-12:
                # This will almost always be true if best_y from offspring; detect by checking if
                # overall best in Y_all equals offspring best.
                if np.isclose(offspring_y[best_off_idx], Y_all[0], rtol=0, atol=1e-12):
                    success_window += 1

            # Adapt at intervals
            if (gen + 1) % max(1, window_size) == 0:
                rate = success_window / max(1, window_size)
                # If success is high, increase exploration; if low, focus.
                if rate >= success_threshold:
                    global_scale = min(2.5, global_scale * 1.1)
                    sigma *= 1.05
                else:
                    global_scale = max(0.25, global_scale * 0.82)
                    sigma *= 0.86
                # Keep sigma within sensible range
                sigma = np.maximum(sigma, 1e-12)
                sigma = np.minimum(sigma, 0.8 * span + 1e-12)
                success_window = 0

            # --- Occasional coordinate refinement around best_x ---
            # Use remaining coordinate budget.
            if coord_budget > 0 and (gen + 1) % coord_every == 0 and coord_tries < 4:
                # Determine coordinate step: proportional to current sigma.
                # Try a subset of coordinates chosen randomly with magnitude bias.
                # Skip very small steps.
                s = sigma / (span + 1e-300)
                # Use bias towards larger sigma (more "uncertain" dimensions)
                weights = np.maximum(s, 1e-12)
                weights = weights / np.sum(weights)

                coords = rng.choice(d, size=min(per_coord_probe, d), replace=False, p=weights)
                # Coordinate probing: for each chosen coordinate, evaluate +/- step
                # but respect remaining budget and coordinate budget.
                # We evaluate at most 2*len(coords) points.
                # Ensure never exceed overall budget.
                # Each probe uses eval_obj which has guard (but we manage to avoid infs).
                step_base = sigma.copy()
                refined = False
                for j in coords:
                    # If budget is almost exhausted, stop probing.
                    if evals >= max_evals or coord_budget <= 0:
                        break
                    # Choose a step along this coordinate
                    step = step_base[j] * (0.5 + 0.5 * rng.random())
                    if step <= 1e-16:
                        continue

                    x_plus = best_x.copy()
                    x_minus = best_x.copy()
                    x_plus[j] = clamp(x_plus[j] + step)[j]
                    x_minus[j] = clamp(x_minus[j] - step)[j]

                    y_plus = eval_obj(x_plus)
                    coord_budget -= 1
                    if y_plus < best_y:
                        best_y = y_plus
                        best_x = x_plus.copy()
                        refined = True

                    if evals >= max_evals or coord_budget <= 0:
                        break

                    y_minus = eval_obj(x_minus)
                    coord_budget -= 1
                    if y_minus < best_y:
                        best_y = y_minus
                        best_x = x_minus.copy()
                        refined = True

                coord_tries += 1
                # If refinement found a better point, inject it into population
                if refined:
                    # Replace worst individual with best_x
                    X[-1] = best_x.copy()
                    Y[-1] = best_y
                    X, Y = sort_population(X, Y)

                # If overall budget is exhausted, stop early.
                if evals >= max_evals:
                    break

        return best_x, best_y

    @staticmethod
    def _get_bounds(func: Any) -> Tuple[Any, Any]:
        # Priority: func.lower/func.upper, else func.bounds.lb/func.bounds.ub
        if hasattr(func, "lower") and hasattr(func, "upper"):
            return getattr(func, "lower"), getattr(func, "upper")
        if hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            if hasattr(b, "lb") and hasattr(b, "ub"):
                return getattr(b, "lb"), getattr(b, "ub")
        # Some harnesses use func.bounds as tuple/list
        if hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            if isinstance(b, (tuple, list)) and len(b) == 2:
                return b[0], b[1]
        raise AttributeError(
            "Bounds not found. Expected func.lower/func.upper or func.bounds.lb/func.bounds.ub."
        )
