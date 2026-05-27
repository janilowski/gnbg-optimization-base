# ALGORITHM_ANALYSIS_NOTE_BEGIN
# Summary: Implements a compact black-box minimization strategy combining
# coordinate-wise random search, a local pattern search (coordinate directions),
# and a simple surrogate-free “informed restart” mechanism. It works for
# arbitrary dimensions and only assumes function evaluations.
#
# Search state: Maintains the current best solution x_best with value f_best,
# a step size sigma that shrinks on progress and grows slightly on stagnation,
# and the remaining evaluation budget.
#
# Candidate generation: At each iteration, generates candidates around x_best
# using (1) random isotropic Gaussian perturbations and (2) deterministic
# coordinate perturbations (plus/minus along randomly permuted axes). Also
# samples a small set of “escape” points when no improvement is seen.
#
# Selection and replacement: Evaluates candidates, keeps the best found, and
# replaces x_best if any candidate improves the current best. If no
# improvement occurs, sigma is increased modestly to encourage exploration.
#
# Adaptation: Uses multiplicative step-size adaptation: sigma decays when
# improvements occur, and increases on stagnation (with caps based on the
# variable scales).
#
# Exploration mechanisms: Uses Gaussian perturbations and occasional random
# restart-like sampling around the best (escape) when stuck.
#
# Exploitation mechanisms: Uses a coordinate pattern search centered at the best
# and refines sigma when a nearby improvement is found.
#
# Boundary handling: Reads bounds from func.lower/func.upper or
# func.bounds.lb/func.bounds.ub, and clamps every candidate to [lb, ub].
# If bounds are invalid (missing or inconsistent), it falls back to using
# finite ranges derived from x_best and a conservative default.
#
# Budget strategy: Strictly enforces the provided evaluation budget by tracking
# evaluations and never performing more than allowed. The algorithm stops when
# the budget is exhausted.
#
# Closest known influences: Heuristic mixture of (a) CMA-inspired step-size
# control without covariance adaptation and (b) coordinate pattern search
# refinement and random restarts.
#
# Novelty or unusual aspects: Very compact, dimension-agnostic approach that
# blends stochastic and deterministic local moves with simple stagnation logic,
# designed to be robust under different objective scales.
#
# Failure modes: If the objective is extremely noisy, the algorithm may
# overreact to false improvements or stagnation. If bounds are very tight,
# clamping can reduce effective search variety, potentially slowing progress.
# ALGORITHM_ANALYSIS_NOTE_END

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np


class Algorithm:
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func) -> Tuple[np.ndarray, float]:
        dim = self.dim
        budget = self.budget
        if budget <= 0:
            # No evaluations allowed; return a deterministic default.
            x0 = np.zeros(dim, dtype=float)
            return x0, float("inf")

        lb, ub = self._read_bounds(func, dim)
        rng = np.random

        evals = 0

        def clamp(x: np.ndarray) -> np.ndarray:
            if lb is not None and ub is not None:
                return np.minimum(np.maximum(x, lb), ub)
            return x

        def evaluate(x: np.ndarray) -> float:
            nonlocal evals
            if evals >= budget:
                return float("inf")
            x = np.asarray(x, dtype=float)
            y = float(func(x))
            evals += 1
            return y

        # Initialize x_best
        if lb is not None and ub is not None:
            x_best = rng.uniform(lb, ub)
        else:
            # Fallback: try a standard normal starting point.
            x_best = rng.normal(size=dim)

        x_best = clamp(x_best)
        f_best = evaluate(x_best)

        # Initialize sigma based on bounds or a default scale.
        if lb is not None and ub is not None:
            span = (ub - lb)
            # Ensure non-zero span to avoid sigma=0.
            span = np.where(np.isfinite(span), span, 1.0)
            span = np.maximum(span, 1e-12)
            sigma = 0.3 * float(np.mean(span))
        else:
            sigma = 0.5

        sigma = max(float(sigma), 1e-12)

        # Helper: propose candidate using isotropic Gaussian perturbation.
        def propose_gaussian(center: np.ndarray, s: float) -> np.ndarray:
            z = rng.normal(size=dim)
            x = center + s * z
            return clamp(x)

        # Helper: coordinate pattern proposal.
        def propose_coordinate_pattern(center: np.ndarray, s: float) -> list:
            # Try a small pattern: +/- along randomly permuted axes.
            axes = rng.permutation(dim)
            k = min(dim, 2 + dim // 2)  # keep candidate count manageable
            candidates = []
            for i in range(k):
                a = int(axes[i])
                ei = np.zeros(dim, dtype=float)
                ei[a] = 1.0
                candidates.append(clamp(center + s * ei))
                candidates.append(clamp(center - s * ei))
            return candidates

        # How many candidates per outer iteration (bounded so we respect budget).
        # We keep it small to stay robust and budget-safe.
        base_batch = 2 + max(1, dim // 2)

        no_improve_streak = 0

        while evals < budget:
            # Determine candidate count while respecting remaining budget.
            remaining = budget - evals
            batch = min(base_batch, remaining)

            candidates = []

            # Always include one Gaussian move.
            candidates.append(propose_gaussian(x_best, sigma))

            # Include a bit of local coordinate exploitation.
            # Use a fraction of remaining budget.
            coord_quota = min(max(0, batch - 1), max(2, dim // 3))
            if coord_quota > 0:
                coord_cands = propose_coordinate_pattern(x_best, sigma)
                if coord_cands:
                    candidates.extend(coord_cands[:coord_quota])

            # Fill remaining with extra Gaussian points.
            while len(candidates) < batch and evals < budget:
                candidates.append(propose_gaussian(x_best, sigma))

            # Evaluate candidates and select best.
            f_local_best = f_best
            x_local_best = x_best
            for x in candidates:
                if evals >= budget:
                    break
                y = evaluate(x)
                if y < f_local_best:
                    f_local_best = y
                    x_local_best = x

            # Adapt based on progress.
            if f_local_best < f_best:
                x_best = x_local_best
                f_best = f_local_best
                no_improve_streak = 0
                # Successful: contract sigma to exploit.
                sigma *= 0.85
            else:
                no_improve_streak += 1
                # Stagnation: expand sigma a bit.
                sigma *= 1.12

            # Escape mechanism when stuck: sample a few random points near bounds center.
            # Keep it budget-aware by only doing this if enough budget remains.
            if no_improve_streak >= 3 and evals < budget:
                remaining = budget - evals
                # Use at most 2*dim proposals, but budget safe.
                esc_n = min(2 + dim // 2, remaining)
                # Center for escape: mid-point of bounds if available, else current best.
                if lb is not None and ub is not None:
                    mid = (lb + ub) * 0.5
                    # Sample around mid with sigma-scaled variance.
                    esc_sigma = sigma * (0.5 + rng.random())
                    for _ in range(esc_n):
                        x = mid + esc_sigma * rng.normal(size=dim)
                        x = clamp(x)
                        y = evaluate(x)
                        if y < f_best:
                            x_best = x
                            f_best = y
                            no_improve_streak = 0
                            sigma *= 0.85
                            break
                else:
                    for _ in range(esc_n):
                        x = x_best + (sigma * (1.0 + rng.random())) * rng.normal(size=dim)
                        x = clamp(x)
                        y = evaluate(x)
                        if y < f_best:
                            x_best = x
                            f_best = y
                            no_improve_streak = 0
                            sigma *= 0.85
                            break
                # After escape attempt, ensure sigma stays within reasonable range.
                if lb is not None and ub is not None:
                    max_span = float(np.mean((ub - lb))) if np.any(np.isfinite(ub - lb)) else 1.0
                    sigma = min(max(sigma, 1e-12), max(1e-12, 2.0 * max_span))
                else:
                    sigma = min(max(sigma, 1e-12), 1e6)

            # Clamp sigma to reasonable numeric range and, if possible, bound span.
            sigma = max(float(sigma), 1e-12)
            if lb is not None and ub is not None:
                span = (ub - lb)
                span = np.where(np.isfinite(span), span, 1.0)
                span = np.maximum(span, 1e-12)
                sigma = min(sigma, 0.75 * float(np.max(span)))

        return np.asarray(x_best, dtype=float), float(f_best)

    @staticmethod
    def _read_bounds(func, dim: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        lb = ub = None

        # Primary options: func.lower/func.upper
        if hasattr(func, "lower") and hasattr(func, "upper"):
            try:
                lb = np.asarray(getattr(func, "lower"), dtype=float).reshape(-1)
                ub = np.asarray(getattr(func, "upper"), dtype=float).reshape(-1)
            except Exception:
                lb = ub = None

        # Alternative: func.bounds.lb / func.bounds.ub
        if (lb is None or ub is None) and hasattr(func, "bounds"):
            b = getattr(func, "bounds")
            if b is not None and hasattr(b, "lb") and hasattr(b, "ub"):
                try:
                    lb = np.asarray(b.lb, dtype=float).reshape(-1)
                    ub = np.asarray(b.ub, dtype=float).reshape(-1)
                except Exception:
                    lb = ub = None

        # Validate and shape-check.
        if lb is not None and ub is not None:
            if lb.size != dim or ub.size != dim:
                # Try to broadcast/resize if possible; otherwise discard.
                if lb.size == 1:
                    lb = np.full(dim, float(lb[0]), dtype=float)
                if ub.size == 1:
                    ub = np.full(dim, float(ub[0]), dtype=float)
            if lb.size == dim and ub.size == dim:
                # If some bounds are inverted or non-finite, allow clamping but
                # correct mild issues by swapping where needed.
                lb2 = lb.copy()
                ub2 = ub.copy()
                mask = np.isfinite(lb2) & np.isfinite(ub2) & (lb2 > ub2)
                if np.any(mask):
                    tmp = lb2[mask].copy()
                    lb2[mask] = ub2[mask]
                    ub2[mask] = tmp
                # Replace non-finite with +/-1e3 scale based on finite counterparts.
                finite = np.isfinite(lb2) | np.isfinite(ub2)
                if not np.any(finite):
                    return None, None
                scale = float(np.mean(np.abs(np.where(np.isfinite(lb2), lb2, ub2))[finite]))
                scale = max(scale, 1.0)
                lb2 = np.where(np.isfinite(lb2), lb2, -1e3 * scale)
                ub2 = np.where(np.isfinite(ub2), ub2, 1e3 * scale)
                return lb2, ub2

        return None, None
