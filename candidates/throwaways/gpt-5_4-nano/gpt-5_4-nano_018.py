# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: This module implements a compact black-box minimization algorithm (box-constrained)
# using a mixture of coordinate-wise random search, optional mirrored sampling, and a simple
# local refinement around the current best. It is designed to work for any dimension and to
# stop exactly when the given evaluation budget is exhausted.
# Search state: The algorithm maintains the current best point x_best and its objective value
# y_best. It also tracks how many evaluations remain and a moving step size sigma that is
# reduced when improvements are found.
# Candidate generation: Each iteration samples several candidate points around x_best by
# drawing random direction vectors with normal distribution, scaling by sigma, and adding to
# x_best. Additionally, it includes coordinate-wise perturbations (1-D steps along a few
# randomly chosen coordinates) to help escape flat regions. Candidates may also use mirrored
# sampling (x_best + d*sigma and x_best - d*sigma) to increase the chance of improvement.
# Selection and replacement: All candidates created in an iteration are evaluated (without
# exceeding budget). The best candidate replaces x_best if it improves y_best.
# Adaptation: The step size sigma decreases when improvements happen and increases slightly
# (or stays larger) when no improvements are found, balancing exploration/exploitation.
# Exploration mechanisms: Random Gaussian directions, coordinate-wise perturbations, and
# occasional larger "global" jumps toward random points within the bounds (when progress
# stalls) encourage exploration.
# Exploitation mechanisms: Candidates are primarily centered at x_best with decreasing sigma,
# and after a period of stagnation the algorithm performs a stronger local refinement by
# shrinking sigma and using mirrored samples more aggressively.
# Boundary handling: Every candidate is projected back into the feasible box via clipping.
# Budget strategy: The algorithm carefully counts evaluations; each objective call decrements
# the remaining budget by exactly one. It never calls the objective more times than allowed.
# Closest known influences: The design loosely resembles evolutionary strategy / CMA-ES-like
# random search with a 1-point state and step-size adaptation, but implemented in a simpler,
# deterministic-budget way suitable for a benchmark harness.
# Novelty or unusual aspects: It combines ES-style sigma adaptation with lightweight
# coordinate-wise moves and a mirrored sampling option, all while using a strict evaluation
# budget accounting and projection to bounds.
# Failure modes: On highly deceptive landscapes or when the optimum lies on a narrow boundary,
# repeated projection/clipping can reduce effective search; also, very small budgets may not
# allow meaningful refinement beyond initial probing.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

from typing import Callable, Optional, Tuple
import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func) -> Tuple[np.ndarray, float]:
        rng = np.random.default_rng()  # harness sets global seed; we still use numpy RNG normally

        lb, ub = self._get_bounds(func, self.dim)
        lb = lb.astype(float)
        ub = ub.astype(float)

        # Ensure lb < ub; if equal, the dimension is fixed.
        widths = ub - lb
        fixed_mask = widths <= 0.0
        widths = np.where(fixed_mask, 1.0, widths)  # avoid division issues

        def clip_to_bounds(x: np.ndarray) -> np.ndarray:
            # Fast projection
            return np.minimum(ub, np.maximum(lb, x))

        evals_used = 0
        remaining = self.budget

        def eval_obj(x: np.ndarray) -> float:
            nonlocal evals_used, remaining
            if remaining <= 0:
                # Should never happen; defensive programming.
                return float("inf")
            y = float(func(x))
            evals_used += 1
            remaining -= 1
            return y

        # Initial point: random within bounds
        x0 = lb + rng.random(self.dim) * (ub - lb)
        x0 = clip_to_bounds(x0)

        y_best = eval_obj(x0)
        x_best = x0.copy()

        # Step size: start proportional to box size
        sigma = 0.25 * np.mean(widths)
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        # Tuning knobs chosen to be robust across dimensions/budgets
        # We cap candidates per batch to avoid overshooting budget with varying dims.
        base_batch = 6 + min(20, self.dim)  # grows moderately with dimension
        stagnation = 0
        best_impr = 0.0

        # If budget is tiny, just return initial best.
        if self.budget <= 1:
            return x_best, y_best

        # Main loop: each loop evaluates up to a batch of candidates.
        while remaining > 0:
            # Dynamically choose how many candidates we can evaluate this round
            # leaving at least 0 evaluations (exact accounting).
            batch = min(base_batch, remaining)

            # Exploration vs exploitation balance (decays with time and stagnation)
            progress = evals_used / max(1, self.budget)
            explore_prob = float(np.clip(0.45 - 0.35 * progress + 0.10 * stagnation, 0.05, 0.60))

            # Occasionally do a stronger local refinement if stuck
            local_refine = (stagnation >= 6) and (remaining >= max(4, self.dim // 2 + 2))
            if local_refine:
                sigma_round = sigma * 0.5
            else:
                sigma_round = sigma

            candidates = []
            # We'll try to fill batch with a mix:
            # - mirrored Gaussian steps centered at x_best
            # - coordinate-wise steps
            # - occasional global random jump
            # We may not generate exactly batch due to budget rounding; that's fine.

            # Determine counts
            n_global = 1 if (rng.random() < explore_prob and remaining > 1) else 0
            n_coord = min(batch, max(0, (batch // 4)))
            n_gauss = max(0, batch - n_global - n_coord)

            # Global random candidate(s) (larger moves)
            for _ in range(n_global):
                x_r = lb + rng.random(self.dim) * (ub - lb)
                x_r = clip_to_bounds(x_r)
                candidates.append(x_r)

            # Coordinate-wise perturbations
            if n_coord > 0:
                # Choose unique-ish coordinates
                # If many fixed dims, still safe: steps there do nothing after clipping.
                coord_idx = rng.integers(0, self.dim, size=n_coord)
                # Each candidate moves along one coordinate with a normal-scaled step
                for j in coord_idx:
                    step = rng.normal() * (0.5 * widths[j])  # coordinate-scale by its box width
                    x_c = x_best.copy()
                    x_c[j] = x_c[j] + step * (0.35 + 0.65 * (1.0 - progress))
                    candidates.append(clip_to_bounds(x_c))

            # Gaussian direction candidates (optionally mirrored)
            # Generate in pairs when possible for mirroring.
            i = 0
            while i < n_gauss:
                # Direction
                d = rng.normal(size=self.dim)
                dn = np.linalg.norm(d)
                if dn == 0:
                    d = np.ones(self.dim)
                    dn = np.sqrt(self.dim)
                d = d / dn

                step_scale = sigma_round * (0.5 + rng.random() * 1.5)
                x_p = clip_to_bounds(x_best + d * step_scale)

                candidates.append(x_p)

                i += 1
                # Mirror with some probability if we still have room.
                if i < n_gauss and rng.random() < 0.55:
                    x_m = clip_to_bounds(x_best - d * step_scale)
                    candidates.append(x_m)
                    i += 1

            # Trim to exactly batch (budget permitting)
            if len(candidates) > batch:
                candidates = candidates[:batch]

            # Evaluate candidates and keep the best
            improved = False
            for x_c in candidates:
                y_c = eval_obj(x_c)
                if y_c < y_best:
                    # Update best
                    improved = True
                    y_gap = y_best - y_c
                    best_impr = max(best_impr, y_gap)
                    y_best = y_c
                    x_best = x_c.copy()

                # Early exit if we have no remaining evaluations
                if remaining <= 0:
                    break

            # Adapt sigma based on progress
            if improved:
                stagnation = 0
                # Reduce sigma more when we see larger improvements
                # Use a smooth factor to avoid collapsing too fast.
                # best_impr helps set scale: if improvement is tiny, reduce less.
                rel = best_impr / (abs(y_best) + 1e-12)
                if rel > 0:
                    factor = float(np.clip(0.85 - 0.25 * np.tanh(rel), 0.65, 0.90))
                else:
                    factor = 0.80
                sigma = max(sigma * factor, 1e-12)
            else:
                stagnation += 1
                # If stuck, slightly increase sigma to escape; cap it by box size.
                sigma *= float(np.clip(1.06 + 0.03 * stagnation, 1.02, 1.20))
                sigma_max = 0.75 * np.mean(widths)
                sigma = min(sigma, sigma_max if sigma_max > 0 else sigma)

            # If sigma becomes extremely small across most dims, re-expand a bit
            # to avoid numerical stagnation.
            if sigma < 1e-14:
                sigma = 0.05 * np.mean(widths)
                stagnation = min(stagnation, 3)

        return x_best, y_best

    @staticmethod
    def _get_bounds(func, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        # Prefer func.lower/func.upper
        if hasattr(func, "lower") and hasattr(func, "upper"):
            lb = np.asarray(func.lower, dtype=float)
            ub = np.asarray(func.upper, dtype=float)
        elif hasattr(func, "bounds"):
            b = func.bounds
            lb = np.asarray(getattr(b, "lb"), dtype=float)
            ub = np.asarray(getattr(b, "ub"), dtype=float)
        else:
            raise AttributeError(
                "Objective function must provide bounds via func.lower/func.upper "
                "or func.bounds.lb/func.bounds.ub."
            )

        if lb.shape == () or ub.shape == ():
            lb = np.full(dim, float(lb))
            ub = np.full(dim, float(ub))
        else:
            lb = lb.reshape(-1).astype(float)
            ub = ub.reshape(-1).astype(float)
            if lb.size != dim or ub.size != dim:
                raise ValueError(f"Bounds size mismatch: expected dim={dim}, got lb={lb.size}, ub={ub.size}")

        # Defensive: ensure proper ordering (swap if needed)
        lo = np.minimum(lb, ub)
        hi = np.maximum(lb, ub)
        return lo, hi
